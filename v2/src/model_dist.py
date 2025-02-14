# -*- coding: utf-8 -*-
import torch
from torch import optim as optim
from torch.nn.modules import Module
from torch.utils.data.dataloader import DataLoader
from private_modules.Torch import model_tra_dist
from typing import Any, Callable
import torch.distributed as dist
from tqdm import tqdm
# from torch.distributed.algorithms.join import Join

class ModelTrainRNN(model_tra_dist.ModelTrain):
    def model_train(self,
                    epoch: int,
                    model: Module,
                    train_loader: DataLoader,
                    optimizer: Any,
                    scheduler,) -> None:
        model.train()
        loss_fn = self.loss_fn
        train_steps = len(train_loader) + (train_loader.num_workers - 1)
        train_ds_size = len(train_loader.dataset)
        world_rank = dist.get_rank()
        # if world_rank == 0:
        train_bar = tqdm(total=train_steps,
                            desc=f"Rank {world_rank} epoch: {epoch} / {self.num_epochs} training")
        current_steps = train_ds_size * (epoch - 1)  # epoch start from 1.
        ddp_loss = torch.zeros(2).cuda()
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
                ddp_loss[0] += loss.detach().item()
                ddp_loss[1] += 1
                current_steps += X.size(0)
                # if world_rank == 0:
                train_bar.update()
                if scheduler is not None:
                    scheduler.step()
        # if world_rank == 0:
        train_bar.close()
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        tra_loss = ddp_loss[0] / ddp_loss[1]
        tra_loss = tra_loss.item()
        return tra_loss
    
    # @torch.no_grad()
    def eval(self, epoch, model: Module, val_data_loader) -> None:
        model.eval()
        loss_fn = self.loss_fn
        world_rank = dist.get_rank()
        val_steps = len(val_data_loader) + (val_data_loader.num_workers - 1)
        # if world_rank == 0:
        val_bar = tqdm(
            total=val_steps,
            desc=f"Rank {world_rank} epoch: {epoch} / {self.num_epochs} validating")
        ddp_loss = torch.zeros(2).cuda()
        with model.join():
            for data in val_data_loader:
                X, Y = data[0], data[1]
                Y_len, Y_flags = data[2], data[3]
                X, Y = X.float().cuda(), Y.float().cuda(),
                Y_len, Y_flags = Y_len.int().cuda(), Y_flags.int().cuda()
                loss = loss_fn(model, X, Y, Y_len, Y_flags,
                            None,
                            **self.kwargs)
                # here is the point, join should combine with backward()
                # loss.backward()
                ddp_loss[0] += loss.detach().item()
                ddp_loss[1] += 1
                # if world_rank == 0:
                val_bar.update()
                # if world_rank == 0:
            val_bar.close()
        # val_loss = val_loss / val_input_num
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / ddp_loss[1]
        val_loss = val_loss.item()
        return val_loss
