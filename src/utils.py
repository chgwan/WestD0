# -*- coding: utf-8 -*-
import os
import math
import shutil
import random
import yaml
import h5py
import numpy as np
import torch
from torch import nn


# ── IO utilities ──────────────────────────────────────────────────────

def load_yaml_config(file_path):
    with open(file_path, 'r') as stream:
        return yaml.safe_load(stream)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def save_to_file(file_name, node_name_data, is_overwrite=False):
    mode = 'w' if is_overwrite else 'a'
    with h5py.File(file_name, mode=mode) as hf:
        keys = list(hf.keys())
        for node_name, node_data in node_name_data.items():
            if node_name in keys:
                del hf[node_name]
            if node_data is not None:
                hf.create_dataset(node_name, data=node_data)
            else:
                hf.create_dataset(node_name, data=h5py.Empty("f"))


def screen_print(src_str: str, width: int = 68, char_fill="="):
    src_str = f" {src_str.strip()} "
    print(src_str.center(width, char_fill))


def clean_dir(train_base_dir):
    shutil.rmtree(train_base_dir)
    os.mkdir(train_base_dir)


# ── Reproducibility ──────────────────────────────────────────────────

def random_init(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ── Masking & Loss ───────────────────────────────────────────────────

def sequence_mask(X, valid_len, value=0):
    if valid_len is not None:
        maxlen = X.size(1)
        mask = torch.arange(maxlen, dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
    return X


class MaskedMSELoss(nn.MSELoss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def forward(self, pred, label, valid_len, valid_label=None):
        assert label.device == pred.device
        weights_len = torch.ones_like(label)
        weights_len = sequence_mask(weights_len, valid_len)
        if valid_label is None:
            weights_label = torch.tensor(1, device=pred.device)
        else:
            weights_label = valid_label[:, None, :]
        self.reduction = 'none'
        unweighted_loss = super().forward(pred, label)
        weighted_loss = (unweighted_loss * weights_len * weights_label).sum(dim=1) / valid_len[:, None]
        return weighted_loss.mean()


# ── Loss functions for training ──────────────────────────────────────

def calc_loss_RNN(
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    Y_len,
    batch_output_flags,
    current_steps=None,
    **kwargs,
):
    step_size = kwargs['step_size']
    window_size = kwargs['window_size']
    seq_len = torch.max(Y_len).item()
    slice_num = seq_len // step_size
    loss_fn = MaskedMSELoss(reduction='mean')
    hidden = None

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
        if torch.isnan(loss).any().item():
            # Yield zero-gradient loss that still goes through the model
            # (needed for DDP gradient sync across ranks)
            loss = (output * 0).sum()
        yield loss


def calc_loss_Former(
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    Y_len,
    batch_output_flags,
    current_steps=None,
    **kwargs,
):
    step_size = kwargs['step_size']
    window_size = kwargs['window_size']
    seq_len = torch.max(Y_len).item()
    slice_num = seq_len // step_size
    loss_fn = MaskedMSELoss(reduction='mean')

    for i in range(slice_num):
        start_idx = i * step_size
        end_length = start_idx + window_size
        X_cut = X[:, start_idx:end_length, :]
        Y_cut = Y[:, start_idx:end_length, :]
        output = model(X_cut)
        dummy_len = Y_len - start_idx
        dummy_len = torch.where(dummy_len < 0, 0, dummy_len)
        loss = loss_fn(Y_cut, output, dummy_len, batch_output_flags)
        if torch.isnan(loss).any().item():
            # Yield zero-gradient loss that still goes through the model
            # (needed for DDP gradient sync across ranks)
            loss = (output * 0).sum()
        yield loss


# ── Inference function ───────────────────────────────────────────────

def inference_fn(
    model,
    X: torch.Tensor,
    Y_tgt: torch.Tensor,
    Y_len,
    batch_output_flags,
    infos,
    **kwargs,
):
    step_size = kwargs['step_size']
    window_size = kwargs['window_size']
    hat_data_dir = kwargs['hat_data_dir']
    device = X.device
    seq_len = torch.max(Y_len).item()
    slice_num = math.ceil(seq_len / step_size)
    res_path = os.path.join(hat_data_dir, 'result.csv')
    loss_fn = MaskedMSELoss(reduction='mean')
    mean = torch.from_numpy(kwargs['mean']).float().to(device)
    stDev = torch.from_numpy(kwargs['stDev']).float().to(device)

    loss_accum = torch.tensor(0.0, device=device)
    col_times = torch.zeros(Y_tgt.size(0), Y_tgt.size(1), device=device)
    col_Y_hat = torch.zeros_like(Y_tgt, device=device)

    for i in range(slice_num):
        padded_hat = torch.zeros_like(Y_tgt)
        valid_hat_len = torch.zeros(Y_tgt.size(0), Y_tgt.size(1), device=device)
        start_idx = i * step_size
        end_length = min(start_idx + window_size, seq_len)
        X_cut = X[:, start_idx:end_length, :]
        Y_cut = Y_tgt[:, start_idx:end_length, :]
        output = model(X_cut)

        padded_hat[:, start_idx:end_length, :] = output
        valid_hat_len[:, start_idx:end_length] = torch.ones(output.size(0), output.size(1))
        col_Y_hat += padded_hat
        col_times += valid_hat_len

        dummy_len = Y_len - start_idx
        dummy_len = torch.where(dummy_len < 0, 0, dummy_len)
        batch_loss = loss_fn(Y_cut, output, dummy_len, batch_output_flags)
        loss_accum += batch_loss
        if torch.isnan(batch_loss).any().item():
            raise ValueError(f"Nan in loss, start_idx: {start_idx}, seq_len: {seq_len}")

    Y_hat = col_Y_hat / col_times[..., None]
    Y_hat = Y_hat * stDev + mean
    Y_tgt = Y_tgt * stDev + mean
    Y_hat = Y_hat.cpu().numpy()
    Y_tgt = Y_tgt.cpu().numpy()
    Y_len = Y_len.cpu().numpy()
    batch_size = Y_tgt.shape[0]
    stDev_arr = stDev.cpu().numpy()

    for idx in range(batch_size):
        info = infos[idx]
        file_name = info['file_name']
        tail_name = os.path.basename(file_name)
        h5_len = Y_len[idx]
        h5_file = os.path.join(hat_data_dir, tail_name)
        h5_dict = {'Y_hat': Y_hat[idx, :h5_len, ...],
                    'Y_tgt': Y_tgt[idx, :h5_len, ...]}
        save_to_file(h5_file, h5_dict)
        loss = np.mean(((h5_dict['Y_hat'] - h5_dict['Y_tgt']) / stDev_arr) ** 2)
        with open(res_path, 'a') as f:
            f.write(f"{loss:.5f}, {tail_name}\n")

    return loss_accum / slice_num
