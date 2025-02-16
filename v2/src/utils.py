# -*- coding: utf-8 -*-
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import torch
from private_modules.Torch import tools
from private_modules.utilities import save_to_file
import h5py
import pickle

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
    loss = loss_fn(Y_hat, Y, None, None)
    # print(loss)
    # print(X.shape)
    # print(Y)
    if torch.isnan(loss).any().item():
        raise ValueError(f"Nan in loss")
    return loss

def inference_fn_MLP(
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
    loss_fn =  tools.MaskedMSELoss(reduction='none')
    mean = kwargs['mean']
    stDev = kwargs['stDev']
    mean = torch.from_numpy(mean).float().cuda()
    stDev = torch.from_numpy(stDev).float().cuda()
    try:
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

        h5_file = os.path.join(hat_data_dir, tail_name)
        h5_dict = dict({})
        h5_dict['Y_hat'] = Y_hat[idx, ...]
        h5_dict['Y_tgt'] = Y_tgt[idx, ...]
        save_to_file(h5_file, h5_dict)
    return batch_loss_mean