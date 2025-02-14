# -*- coding: utf-8 -*-
import torch
from torch import optim as optim
from torch.nn.modules import Module
from torch.utils.data.dataloader import DataLoader
from private_modules.Torch import model_tra_tune
from typing import Any, Callable
from ray import train
from tqdm import tqdm
# from torch.distributed.algorithms.join import Join
import torch.distributed as dist
from private_modules import screen_print


class ModelTuneRNN(model_tra_tune.ModelTune):
    def model_train(self,
                    epoch: int,
                    model: Module,
                    train_loader: DataLoader,
                    optimizer: Any,
                    scheduler: Any) -> None:
        loss_fn = self.loss_fn
        train_steps = len(train_loader) + (train_loader.num_workers - 1)
        train_ds_size = len(train_loader.dataset)
        model.train()
        world_rank = train.get_context().get_world_rank()
        # if world_rank == 0:
        train_bar = tqdm(
            total=train_steps,
            desc=f"Rank {world_rank} epoch: {epoch} / {self.num_epochs} training")
        # tra_loss = 0
        # input_num = 0
        ddp_loss = torch.zeros(2).cuda()
        current_steps = train_ds_size * (epoch - 1)  # epoch start from 1.
        # with Join([model]):
        with model.join():
            for data in train_loader:
                optimizer.zero_grad(set_to_none=True)
                X, Y = data[0], data[1]
                Y_len, Y_flags = data[2], data[3]
                X, Y = X.float().cuda(), Y.float().cuda(),
                Y_len, Y_flags = Y_len.int().cuda(), Y_flags.int().cuda()
                loss = loss_fn(model, X, Y, Y_len, Y_flags,
                               current_steps,
                               **self.kwargs)
                loss.backward()
                optimizer.step()
                # tra_loss += loss.detach().item()
                # input_num += X.size(0)
                # input_num += 1
                ddp_loss[0] += loss.detach().item()
                ddp_loss[1] += 1
                current_steps += X.size(0)
                # if world_rank == 0:
                train_bar.update()
                if scheduler is not None:
                    scheduler.step()                
        # if world_rank == 0:
        train_bar.close()
        # tra_loss = tra_loss / input_num
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        tra_loss = ddp_loss[0] / ddp_loss[1]
        tra_loss = tra_loss.item()
        return tra_loss

    # @torch.no_grad()
    def eval(self, epoch, model: Module, val_data_loader) -> None:
        loss_fn = self.loss_fn
        world_rank = train.get_context().get_world_rank()
        val_steps = len(val_data_loader) + (val_data_loader.num_workers - 1)
        model.eval()
        # if world_rank == 0:
        val_bar = tqdm(
            total=val_steps,
            desc=f"Rank {world_rank} epoch: {epoch} / {self.num_epochs} validating")
        ddp_loss = torch.zeros(2).cuda()
        # with Join([model]):
        with model.join():
            for data in val_data_loader:
                X, Y = data[0], data[1]
                Y_len, Y_flags = data[2], data[3]
                X, Y = X.float().cuda(), Y.float().cuda(),
                Y_len, Y_flags = Y_len.int().cuda(), Y_flags.int().cuda()
                loss = loss_fn(model, X, Y, Y_len, Y_flags,
                               None,
                               **self.kwargs)
                # here is a trick
                # loss.backward()
                # val_loss += loss.detach().item()
                # val_input_num += X.size(0)
                # val_input_num += 1
                ddp_loss[0] += loss.detach().item()
                ddp_loss[1] += 1
                # if world_rank == 0:
                val_bar.update()
        # if world_rank == 0:
        val_bar.close()
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / ddp_loss[1]
        val_loss = val_loss.item()
        return val_loss


class ModelInferRNN(model_tra_tune.ModelInfer):
    @torch.no_grad()
    def model_infer(self, model, infer_loader, infer_fn, **kwargs):
        model.eval()
        infer_steps = len(infer_loader) + (infer_loader.num_workers - 1)
        world_rank = train.get_context().get_world_rank()
        # if world_rank == 0:
        infer_bar = tqdm(
            total=infer_steps,
                desc=f'Rank {world_rank} model inferring')
        for data in infer_loader:
            X, Y = data[0], data[1]
            batch_len, batch_node_flags = data[2], data[3]
            X, Y = X.float().cuda(), Y.float().cuda()
            batch_len, batch_node_flags = batch_len.int().cuda(), batch_node_flags.int().cuda()
            infos = data[-1]
            infer_fn(model, X, Y, batch_len, batch_node_flags, infos, **kwargs)
            # if world_rank == 0:
            infer_bar.update()
        # if world_rank == 0:
        infer_bar.close()
        return 0


class ModelInferRNNCPU(model_tra_tune.ModelInfer):
    @torch.no_grad()
    def model_infer(self, model, infer_loader, infer_fn, **kwargs):
        model.eval()
        model.cpu()
        infer_steps = len(infer_loader) + (infer_loader.num_workers - 1)
        world_rank = train.get_context().get_world_rank()
        if world_rank == 0:
            infer_bar = tqdm(total=infer_steps,
                             desc=f'model inferring')
        for data in infer_loader:
            X, Y = data[0], data[1]
            batch_len, batch_node_flags = data[2], data[3]
            X, Y = X.float(), Y.float()
            batch_len, batch_node_flags = batch_len.int(), batch_node_flags.int()
            # X, Y = X.float().cuda(), Y.float().cuda()
            # batch_len, batch_node_flags = batch_len.int().cuda(), batch_node_flags.int().cuda()
            infos = data[-1]
            infer_fn(model, X, Y, batch_len, batch_node_flags, infos, **kwargs)
            if world_rank == 0:
                infer_bar.update()
        if world_rank == 0:
            infer_bar.close()
        return 0
