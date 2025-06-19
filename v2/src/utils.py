# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from private_modules.Torch import tools
from private_modules import save_to_file, screen_print
from private_modules import load_yaml_config
import math

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
    step_size = kwargs['step_size']
    window_size = kwargs['window_size']
    hat_data_dir = kwargs['hat_data_dir']
    model_type = kwargs.get('model_type', "Others")
    device = X.device
    seq_len = torch.max(Y_len).item()
    slice_num  = math.ceil(seq_len / step_size)
    hidden = None
    res_path = os.path.join(hat_data_dir, 'result.csv')
    loss_fn = tools.MaskedMSELoss(reduction='mean')
    mean = kwargs['mean']
    stDev = kwargs['stDev']
    mean = torch.from_numpy(mean).float().to(device=device)
    stDev = torch.from_numpy(stDev).float().to(device=device)
    
    # data collect
    loss_accum = torch.tensor(0.0).float().to(device=device)
    col_times = torch.zeros(Y_tgt.size(0), Y_tgt.size(1), device=device)
    col_Y_hat = torch.zeros_like(Y_tgt, device=device)
    for i in range(slice_num):
        padded_hat = torch.zeros_like(Y_tgt)
        valid_hat_len = torch.zeros(Y_tgt.size(0), Y_tgt.size(1), device=device)
        start_idx = i * step_size
        end_length = min(start_idx + window_size, seq_len)
        X_cut = X[:, start_idx:end_length, :]
        Y_cut = Y_tgt[:, start_idx:end_length, :]
        if model_type == "RNN":
            output, hidden = model(X_cut, hidden)
        else:
            output = model(X_cut)

        padded_hat[:, start_idx:end_length, :] = output
        valid_hat_len[:, start_idx:end_length] = torch.ones(output.size(0), 
                                                            output.size(1))
        col_Y_hat = col_Y_hat + padded_hat
        col_times = col_times + valid_hat_len
        dummy_len = Y_len - start_idx
        dummy_len = torch.where(dummy_len < 0, 0, dummy_len)
        batch_loss = loss_fn(Y_cut, output, dummy_len, batch_output_flags)
        # batch_loss = batch_loss.sum(dim=1).sum(dim=-1)
        loss_accum += batch_loss
        if torch.isnan(batch_loss).any().item():
            raise ValueError(f"Nan in loss in {kwargs['infos']}, " 
                    f"start_idx: {start_idx} "
                    f"sequence length: {seq_len} ")
        
    Y_hat = col_Y_hat / col_times[..., None]
    Y_hat = Y_hat * stDev + mean
    Y_tgt = Y_tgt * stDev + mean
    Y_hat = Y_hat.cpu().numpy()
    Y_tgt = Y_tgt.cpu().numpy()
    Y_len = Y_len.cpu().numpy()
    batch_size = Y_tgt.shape[0]

    mean_arr = mean.cpu().numpy()
    stDev_arr = stDev.cpu().numpy()

    for idx in range(batch_size):
        info = infos[idx]
        file_name = info['file_name']
        tail_name = os.path.basename(file_name)
        h5_len = Y_len[idx]
        h5_file = os.path.join(hat_data_dir, tail_name)
        h5_dict = dict({})
        h5_Y_hat = Y_hat[idx, :h5_len, ...]
        h5_Y_tgt = Y_tgt[idx, :h5_len, ...]
        h5_dict['Y_hat'] = h5_Y_hat
        h5_dict['Y_tgt'] = h5_Y_tgt
        save_to_file(h5_file, h5_dict) 
        loss = np.mean(((h5_Y_hat - h5_Y_tgt) / stDev_arr) ** 2) 
        with open(res_path, 'a') as f:
            f.write(f"{loss:.5f}, {tail_name} \n")
    return loss_accum / slice_num

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

def calc_loss_Former(
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
    # hidden = None
    # loss_accum = 0
    # if model.training:
    for i in range(slice_num):
        start_idx = i * step_size
        end_length = start_idx + window_size
        X_cut = X[:, start_idx:end_length, :]
        Y_cut = Y[:, start_idx:end_length, :]
        # output, hidden = model(X_cut, hidden)
        output = model(X_cut)
        # hidden = (hidden[0].detach(), hidden[1].detach())
        dummy_len = Y_len - start_idx
        dummy_len = torch.where(dummy_len < 0, 0, dummy_len)
        loss = loss_fn(Y_cut, output, dummy_len, batch_output_flags)
        # loss_accum += loss            
        if torch.isnan(loss).any().item():
            # block nan join training.
            loss = torch.tensor(1000, requires_grad=True)
            loss.grad = torch.tensor(0.0)
        #     raise ValueError(f"Nan in loss in {kwargs['infos']}, " 
        #                         f"start_idx: {start_idx} "
        #                         f"sequence length: {seq_len} ")
        yield loss

def get_nodes(config_f=None):
    if config_f is None:
        config_f = "$HOME/Papers/WestD0/v2/configs/former.yml"
    config_f = os.path.expandvars(config_f)
    config = load_yaml_config(config_f)
    data_params = config["data"]

    input_list = data_params['input_list']
    output_list = data_params['output_list']
    input_nodes, output_nodes = [], []
    for dummy_list_name in input_list:
        if "_real" in dummy_list_name: 
            dummy_list_name = dummy_list_name[:-5]
            i = 3
        if "_ref" in dummy_list_name: 
            dummy_list_name = dummy_list_name[:-4]
            i = 0
        input_nodes.extend([f"{dummy_node}_{i}" for dummy_node in config['nodes'][dummy_list_name]])
    for dummy_list_name in output_list:
        output_nodes.extend(config['nodes'][dummy_list_name])
    return input_nodes, output_nodes