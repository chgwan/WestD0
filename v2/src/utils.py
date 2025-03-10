# -*- coding: utf-8 -*-
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import torch
from private_modules.Torch import tools
from private_modules.utilities import save_to_file, screen_print
import h5py
import pickle
from torch import nn
from tqdm import tqdm

def calc_loss_MLP(
    model,
    X: torch.tensor,
    Y: torch.tensor,
    Y_len,
    batch_output_flags,
    current_steps = None,
    **kwargs,
):
    loss_fn = tools.MaskedMSELoss(reduction='mean')
    Y_hat = model(X)
    loss = loss_fn(Y_hat, Y, Y_len, batch_output_flags)
    # print(loss)
    # print(X.shape)
    # print(Y)
    if torch.isnan(loss).any().item():
        raise ValueError(f"Nan in loss")
    return loss

def inference_fn(
    model,
    X: torch.tensor,
    Y_tgt: torch.tensor,
    Y_len,
    batch_output_flags,
    infos,
    **kwargs,
):
    hat_data_dir = kwargs['hat_data_dir']
    res_path = os.path.join(hat_data_dir, 'result.csv')
    with open(res_path, 'a') as f:
        f.write(f"loss, file_name \n")
    loss_fn =  tools.MaskedMSELoss(reduction='none')
    mean = kwargs['mean']
    stDev = kwargs['stDev']
    mean = torch.from_numpy(mean).float().cuda()
    stDev = torch.from_numpy(stDev).float().cuda()
    try:
        if kwargs['is_RNN']:
            Y_hat, state = model(X)
        else:
            Y_hat = model(X)
    except Exception as e:
        raise Exception(str(infos[0]), repr(e))
    batch_loss = loss_fn(Y_tgt, Y_hat, Y_len, batch_output_flags)
    batch_loss = batch_loss.sum(dim=1).sum(dim=-1) / (Y_len * batch_output_flags.sum(dim=-1))
    batch_loss_arr = batch_loss.detach().cpu().numpy()
    batch_loss_mean = batch_loss.mean()
    batch_size = batch_loss.shape[0]
    Y_hat = Y_hat * stDev + mean
    Y_tgt = Y_tgt * stDev + mean

    Y_hat = Y_hat.cpu().numpy()
    Y_tgt = Y_tgt.cpu().numpy()
    Y_len = Y_len.cpu().numpy()

    # shot_dict = {
    #     'Y_hat': Y_hat, 
    #     'Y_tgt': Y_tgt,
    #     'Y_len': Y_len.cpu().numpy(),
    #     'batch_output_flags': batch_output_flags.cpu().numpy(),
    #     'infos': infos,
    # }
    # with open('a.b', 'wb') as f:
    #     pickle.dump(shot_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    for idx in range(batch_size):
        info = infos[idx]
        loss = batch_loss_arr[idx]
        file_name = info['file_name']
        tail_name = os.path.basename(file_name)
        with open(res_path, 'a') as f:
            f.write(f"{loss:.5f}, {tail_name} \n")
        
        h5_len = Y_len[idx]
        h5_file = os.path.join(hat_data_dir, tail_name)
        h5_dict = dict({})
        h5_dict['Y_hat'] = Y_hat[idx, :h5_len, ...]
        h5_dict['Y_tgt'] = Y_tgt[idx, :h5_len, ...]
        save_to_file(h5_file, h5_dict)
    return batch_loss_mean

def calc_loss_RNN(
    model,
    X: torch.tensor,
    Y: torch.tensor,
    Y_len,
    batch_output_flags,
    current_steps = None,
    **kwargs,
):
    step_size = kwargs['step_size']
    window_size = kwargs['window_size']
    # accu_steps = kwargs['accumulated_steps']
    seq_len = torch.max(Y_len).item()
    # screen_print(f"sequence length: {seq_len.item()}")
    # slice_num = (seq_len - window_size) // step_size
    slice_num = (seq_len) // step_size
    loss_fn = tools.MaskedMSELoss(reduction='mean')
    # loss_fn = nn.MSELoss(reduction='none')
    hidden = None
    loss_accum = 0
    # if model.training:
    for i in range(slice_num):
        start_idx = i * step_size
        end_length = start_idx + window_size
        X_cut = X[:, start_idx:end_length, :]
        Y_cut = Y[:, start_idx:end_length, :]
        output, hidden = model(X_cut, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())
        dummy_len = Y_len - start_idx
        dummy_len = torch.where(dummy_len < 0, 0, dummy_len)
        loss = loss_fn(Y_cut, output, dummy_len, batch_output_flags)
        loss_accum += loss            
        if torch.isnan(loss).any().item():
            raise ValueError(f"Nan in loss in {kwargs['infos']}, " 
                                f"start_idx: {start_idx} "
                                f"sequence length: {seq_len} ")
        yield loss
        # if ((i + 1) % accu_steps == 0) or (i + 1) == slice_num:
        #     # loss_accum.backward()
        #     yield loss_accum / accu_steps
        #     loss_accum = 0
        # yield loss
    # else:
    #     output, hidden = model(X, hidden)
    #     loss = loss_fn(Y, output, Y_len, batch_output_flags)
    #     yield loss