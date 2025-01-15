# -*- coding: utf-8 -*-
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import torch
from private_modules.Torch import tools

def read_mat_file(mat_file: os.PathLike, nodes):
    data = loadmat(mat_file)
    node_stat_dict = dict({})
    # assume same time axis. 
    Ip_struct = data['Ip_scope'][0,0]
    time_field = Ip_struct['time'].squeeze()
    node_stat_dict['time'] = time_field
    for node in nodes:
        scope_name = node[:-2]
        slice = int(node[-1])
        scope_struct = data[scope_name][0,0]
        # time_field = scope_struct['time'].squeeze()
        # Extract signals and values
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0,0] 
        # num_signals = signal_values.shape[1] 
        signal_data = signal_values[:, slice]
        node_stat_dict[node] = signal_data
    return node_stat_dict


def get_ref_start(mat_file):
    data_dict = read_mat_file(mat_file, 'Ip_scope_0')
    Ip_ref_data = data_dict['Ip_scope_0']
    eps = 1e-5
    set_value = 1.0
    ids = Ip_ref_data > (set_value + eps)
    start_idx = np.min(np.arange(len(Ip_ref_data))[ids])
    start_idx = start_idx - 1
    return start_idx

def get_shots(stat_file):
    df = pd.read_csv(stat_file, index_col=0)
    ids = df.index[df.loc[:, "length"] > 1000]
    selected_shots = ids.to_list()
    return selected_shots
    

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
    loss = loss_fn(Y_hat, Y, None, batch_output_flags)
    if torch.isnan(loss).any().item():
        raise ValueError(f"Nan in loss")
    return loss