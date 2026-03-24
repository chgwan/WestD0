# -*- coding: utf-8 -*-
"""
Distributed training and inference for WEST D0 models.

Compatible with torchrun:
    torchrun --nproc_per_node=4 run.py --config configs/former.yml
"""
import os
import pathlib
import datetime
import threading

import natsort
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import screen_print


# ---------------------------------------------------------------------------
# TensorBoard logger (background thread on rank 0)
# ---------------------------------------------------------------------------

class _TBLogger:
    """Non-blocking TensorBoard logger that runs in a daemon thread."""

    def __init__(self, log_dir):
        self._queue = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def log(self, metrics: dict):
        with self._lock:
            self._queue.append(metrics)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=30)
        self._writer.close()

    def _run(self):
        while not self._stop.is_set():
            self._flush()
            self._stop.wait(timeout=5)
        self._flush()

    def _flush(self):
        with self._lock:
            batch, self._queue = self._queue, []
        for metrics in batch:
            epoch = metrics['epoch']
            for key, val in metrics.items():
                if key != 'epoch':
                    self._writer.add_scalar(key, val, epoch)
        if batch:
            self._writer.flush()


# ---------------------------------------------------------------------------
# Distributed setup helpers (torchrun-compatible)
# ---------------------------------------------------------------------------

def _setup_dist(timeout_s=1800):
    """Initialize the process group from torchrun env vars."""
    if not dist.is_initialized():
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            timeout=datetime.timedelta(seconds=timeout_s),
        )
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def _setup_model(model):
    model.cuda()
    return DDP(model, broadcast_buffers=False, find_unused_parameters=False)


# ---------------------------------------------------------------------------
# Distributed training for truncated-BPTT / Former
# ---------------------------------------------------------------------------

class ModelTrainTruncatedRNN:
    def __init__(self, data_gen, num_epochs, loss_fn,
                 train_base_dir, tra_settings, **kwargs):
        self.data_gen = data_gen
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.train_base_dir = pathlib.Path(train_base_dir)
        self.tra_settings = tra_settings
        self.kwargs = kwargs

    # ── single-epoch train / eval ─────────────────────────────────────

    def _model_train(self, epoch, model, train_loader, optimizer, scheduler):
        model.train()
        loss_fn = self.loss_fn
        train_steps = len(train_loader) + (train_loader.num_workers - 1)
        world_rank = dist.get_rank()
        train_bar = tqdm(total=train_steps,
                         desc=f"Rank {world_rank} epoch: {epoch}/{self.num_epochs} training",
                         disable=(world_rank != 0))
        ddp_loss = torch.zeros(2).cuda()

        for data in train_loader:
            X, Y = data[0].float().cuda(), data[1].float().cuda()
            Y_len, Y_flags = data[2].int().cuda(), data[3].int().cuda()
            infos = data[-1]
            loss_gen = loss_fn(model, X, Y, Y_len, Y_flags, None,
                               infos=infos, **self.kwargs)
            for loss in loss_gen:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ddp_loss[0] += loss.detach().item()
                ddp_loss[1] += 1
            if scheduler is not None:
                scheduler.step()
            train_bar.update()
        train_bar.close()
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        return (ddp_loss[0] / ddp_loss[1]).item()

    def _eval(self, epoch, model, val_loader):
        loss_fn = self.loss_fn
        world_rank = dist.get_rank()
        val_steps = len(val_loader) + (val_loader.num_workers - 1)
        val_bar = tqdm(total=val_steps,
                       desc=f"Rank {world_rank} epoch: {epoch}/{self.num_epochs} validating",
                       disable=(world_rank != 0))
        ddp_loss = torch.zeros(2).cuda()
        # Use unwrapped model for eval — no DDP gradient sync needed
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.eval()

        with torch.no_grad():
            for data in val_loader:
                X, Y = data[0].float().cuda(), data[1].float().cuda()
                Y_len, Y_flags = data[2].int().cuda(), data[3].int().cuda()
                infos = data[4]
                loss_gen = loss_fn(raw_model, X, Y, Y_len, Y_flags,
                                   infos=infos, **self.kwargs)
                for loss in loss_gen:
                    ddp_loss[0] += loss.item()
                    ddp_loss[1] += 1
                val_bar.update()
        val_bar.close()
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        return (ddp_loss[0] / ddp_loss[1]).item()

    # ── epoch loop ────────────────────────────────────────────────────

    def _epoch_train(self, world_rank, epoch, model, optimizer, scheduler,
                     train_loader, val_loader, logger):
        tra_loss = self._model_train(epoch, model, train_loader, optimizer, scheduler)
        model_state = model.state_dict()
        consume_prefix_in_state_dict_if_present(model_state, "module.")
        metrics = {"train_loss": tra_loss, "epoch": epoch}

        eval_interval = self.tra_settings.get('eval_interval_epochs', 1)
        save_ckpt = self.tra_settings.get('save_ckpt', True)
        save_steps = self.tra_settings.get('save_checkpoint_steps', 1)
        keep_max = self.tra_settings.get('keep_checkpoint_max', 10)

        if epoch % eval_interval == 0:
            val_loss = self._eval(epoch, model, val_loader)
            metrics['loss'] = val_loss
            if world_rank == 0:
                model_dir = self.train_base_dir / "Model"
                model_dir.mkdir(exist_ok=True)
                pt_path = model_dir / f"{epoch}-{tra_loss:.6f}-{val_loss:.6f}.pt"
                torch.save(model_state, pt_path)

        if save_ckpt and epoch % save_steps == 0:
            if 'loss' not in metrics:
                val_loss = self._eval(epoch, model, val_loader)
                metrics['loss'] = val_loss
            if world_rank == 0:
                log_dir = self.train_base_dir / "Log"
                ckpt_path = log_dir / f"checkpoint{epoch:06d}.pt"
                ckpt = {'model': model_state, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                if scheduler is not None:
                    ckpt['scheduler'] = scheduler.state_dict()
                torch.save(ckpt, ckpt_path)
                # prune old checkpoints
                ckpt_list = natsort.natsorted(list(log_dir.glob("checkpoint*.pt")))
                while len(ckpt_list) > keep_max:
                    ckpt_list[0].unlink()
                    ckpt_list.pop(0)

        if world_rank == 0 and logger is not None:
            logger.log(metrics)

    # ── entry point (called once per torchrun process) ────────────────

    def run_train(self, model, restore=False):
        world_rank, _ = _setup_dist(
            timeout_s=self.tra_settings.get('timeout', 1800))

        # Prepare data loaders — each rank gets its own split
        loaders = self.data_gen.sp_ratio_wz()
        train_loader = loaders[0][world_rank]
        val_loader = loaders[1][world_rank]

        self.train_base_dir.mkdir(exist_ok=True)
        log_dir = self.train_base_dir / "Log"
        log_dir.mkdir(exist_ok=True)

        # TensorBoard logger on rank 0 only
        logger = _TBLogger(log_dir) if world_rank == 0 else None

        start_epoch = 1
        if restore:
            ckpt_path = os.path.expandvars(self.tra_settings['checkpoint_path'])
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['model'])

        model = _setup_model(model)

        lr = self.tra_settings['learning_rate']
        optim_fn = self.tra_settings.get('optimizer_fn', torch.optim.SGD)
        sched_fn = self.tra_settings.get('scheduler_fn', None)

        optimizer = optim_fn(model.parameters(), lr=lr)
        scheduler = sched_fn(optimizer) if sched_fn is not None else None

        if restore:
            optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler is not None and 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])

        for epoch in range(start_epoch, self.num_epochs + 1):
            self._epoch_train(world_rank, epoch, model, optimizer, scheduler,
                              train_loader, val_loader, logger)

        if logger is not None:
            logger.close()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Distributed inference
# ---------------------------------------------------------------------------

class ModelInferRNN:
    def __init__(self, infer_loaders, infer_fn=None, world_size=1, **kwargs):
        self.infer_loaders = infer_loaders
        self.infer_fn = infer_fn
        self.world_size = world_size
        self.kwargs = kwargs

        hat_data_dir = kwargs['hat_data_dir']
        res_path = os.path.join(hat_data_dir, 'result.csv')
        with open(res_path, 'a') as f:
            f.write("loss, file_name\n")

    @torch.no_grad()
    def _model_infer(self, model, infer_loader):
        model.eval()
        ddp_loss = torch.zeros(2).cuda()
        infer_steps = len(infer_loader) + (infer_loader.num_workers - 1)
        world_rank = dist.get_rank()
        infer_bar = tqdm(total=infer_steps, desc=f'Rank {world_rank} model inferring',
                         disable=(world_rank != 0))

        for data in infer_loader:
            X, Y = data[0].float().cuda(), data[1].float().cuda()
            batch_len = data[2].int().cuda()
            batch_flags = data[3].int().cuda()
            infos = data[-1]
            infer_loss = self.infer_fn(model, X, Y, batch_len, batch_flags, infos,
                                       **self.kwargs)
            ddp_loss[0] += infer_loss.detach().item()
            ddp_loss[1] += 1
            infer_bar.update()
        infer_bar.close()
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        return (ddp_loss[0] / ddp_loss[1]).item()

    def run_infer(self, model):
        world_rank, _ = _setup_dist(timeout_s=180)

        model = _setup_model(model)
        infer_loader = self.infer_loaders[world_rank]
        loss = self._model_infer(model, infer_loader)

        if world_rank == 0:
            screen_print(f"Inference loss: {loss:.6f}")

        dist.destroy_process_group()
        return loss
