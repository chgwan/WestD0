# -*- coding: utf-8 -*-
import math
import os
import random
import numpy as np
import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
from typing import List, LiteralString
import h5py


# ── Data reading ─────────────────────────────────────────────────────

def SMA(data, window_size):
    if window_size <= 0:
        raise ValueError("Window size of SMA must be greater than zero.")
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def read_h5_tokamak(
    h5_file: os.PathLike,
    nodes: List[LiteralString],
    dtype=np.float32,
    **kwargs,
):
    rare_cap = kwargs.get('rare_cap', 1)
    non_filter_nodes = kwargs.get('non_filter_nodes', [])
    filter_func = kwargs['filter_func']
    filter_wz = kwargs['filter_wz']
    half_filter_wz = filter_wz // 2

    with h5py.File(h5_file, mode="r") as hf:
        timeAxis = hf["time"][()]
        IMAS_time = hf['IMAS_time'][()]
        IMAS_start, IMAS_end = IMAS_time[0], IMAS_time[-1]

        ids = (timeAxis >= IMAS_start) & (timeAxis < IMAS_end)
        timeAxis = timeAxis[ids]
        timeAxis = timeAxis[half_filter_wz:-half_filter_wz]

        data = []
        node_flags = []
        for node in nodes:
            if hf[node].shape is None or len(np.unique(hf[node][()])) <= rare_cap:
                nodeData = np.zeros_like(timeAxis, dtype=dtype)
                node_flags.append(0)
            else:
                node_flags.append(1)
                nodeData = hf[node][()][ids]
                inf_value = 3.2e32
                nodeData = np.nan_to_num(nodeData, posinf=inf_value, neginf=-inf_value)
                if node not in non_filter_nodes:
                    nodeData = filter_func(nodeData, filter_wz)
                else:
                    nodeData = nodeData[half_filter_wz:-half_filter_wz]
                assert len(timeAxis) == len(nodeData), \
                    "file: %s, Node:%s" % (h5_file, node)
            if len(nodeData.shape) == 1:
                nodeData = np.array(nodeData)[:, np.newaxis]
            data.append(nodeData)
        node_flags = np.array(node_flags, dtype=int)
        data = np.concatenate(data, dtype=dtype, axis=1)
    return data, node_flags, timeAxis


# ── Dataset ──────────────────────────────────────────────────────────

class _BaseH5(Data.IterableDataset):
    def __init__(self, h5_list, input_nodes, output_nodes, *,
                 read_h5=read_h5_tokamak, dtype=np.float32, **kwargs):
        super().__init__()
        self.h5_list = h5_list
        nodes = []
        nodes.extend(input_nodes)
        nodes.extend(output_nodes)
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.nodes = nodes
        self.read_h5 = read_h5
        self.dtype = dtype
        self.h5s_len = len(h5_list)
        self.kwargs = kwargs

    def __len__(self):
        return self.h5s_len

    def __iter__(self):
        worker_info = Data.get_worker_info()
        if worker_info is None:
            iter_h5s = self.h5_list
        else:
            # Use divmod for balanced splits (matches _evenly_split)
            parts = _evenly_split(self.h5_list, worker_info.num_workers)
            iter_h5s = parts[worker_info.id]
        return self._gen(iter_h5s)

    def _gen(self, iter_h5s):
        for h5_file in iter_h5s:
            data = self.get_h5_data(h5_file)
            data = self.callback(data)
            yield data

    def callback(self, data):
        return data

    def get_h5_data(self, h5_file):
        raise NotImplementedError


class StdWESTShotDS(_BaseH5):
    def get_h5_data(self, h5_file):
        info = {}
        data, node_flags, _ = self.read_h5(h5_file, self.nodes, self.dtype, **self.kwargs)
        shot_len = data.shape[0]
        info['file_name'] = h5_file
        return data, shot_len, node_flags, info

    def callback(self, data):
        MS_df = self.kwargs['MS_df']
        means = MS_df.loc['mean', self.nodes].to_numpy()
        stds = MS_df.loc['stDev', self.nodes].to_numpy()
        XY = (data[0] - means) / stds
        X = XY[..., :-len(self.output_nodes)]
        Y = XY[..., -len(self.output_nodes):]
        Y_len = data[1]
        Y_flags = data[2][-len(self.output_nodes):]
        info = data[-1]
        return X, Y, Y_len, Y_flags, info


# ── Collate ──────────────────────────────────────────────────────────

def pad_seq_collate(batch):
    if len(batch) == 1:
        b = batch[0]
        batch_X = torch.unsqueeze(torch.from_numpy(b[0]), 0)
        batch_Y = torch.unsqueeze(torch.from_numpy(b[1]), 0)
        batch_len = torch.unsqueeze(torch.tensor(b[2], dtype=torch.int32), 0)
        batch_flags = torch.unsqueeze(torch.tensor(b[3], dtype=torch.int32), 0)
        batch_info = [b[4]]
    else:
        Xs, Ys, lens, flags, infos = [], [], [], [], []
        for s in batch:
            Xs.append(torch.from_numpy(s[0]))
            Ys.append(torch.from_numpy(s[1]))
            lens.append(torch.tensor(s[2], dtype=torch.int32))
            flags.append(torch.tensor(s[3], dtype=torch.int32))
            infos.append(s[4])
        batch_X = pad_sequence(Xs, batch_first=True)
        batch_Y = pad_sequence(Ys, batch_first=True)
        batch_len = torch.tensor(lens, dtype=torch.int32)
        batch_flags = torch.stack(flags)
        batch_info = infos

    return (batch_X.float(), batch_Y.float(),
            batch_len.int(), batch_flags.int(), batch_info)


# ── Data splitting ───────────────────────────────────────────────────

def _train_split(src, split_ratio):
    n = len(src)
    if isinstance(split_ratio, float):
        avg = (1 - split_ratio) / 2
        split_ratio = [split_ratio, avg, avg]
    r0 = split_ratio[0] / sum(split_ratio)
    r1 = split_ratio[1] / sum(split_ratio)
    d0 = int(n * r0)
    d1 = int(n * (r0 + r1))
    return src[:d0], src[d0:d1], src[d1:]


def _evenly_split(src, n):
    """Split *src* into *n* parts of equal length (last part may be shorter by at most 1)."""
    q, r = divmod(len(src), n)
    result, start = [], 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        result.append(src[start:start + size])
        start += size
    return result


# ── DataLoader generator ─────────────────────────────────────────────

class H5GenDataLoader:
    def __init__(self, h5s, input_nodes, output_nodes, batch_size=1,
                 num_workers=1, world_size=1, DS=StdWESTShotDS,
                 MS_df=None, stat_df=None, pin_memory=False, **kwargs):
        if batch_size % world_size != 0:
            raise ValueError(f"batch_size {batch_size} must be divisible by world_size {world_size}")
        self.h5s = h5s
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.batch_size = batch_size // world_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.DS = DS
        self.pin_memory = pin_memory
        self.split_ratio = [0.8, 0.1, 0.1]
        self.kwargs = kwargs
        self.kwargs['MS_df'] = MS_df
        self.kwargs['stat_df'] = stat_df

    def set_split_ratio(self, ratio):
        self.split_ratio = ratio

    def _make_dl(self, iter_h5s):
        ds = self.DS(
            iter_h5s,
            self.input_nodes,
            self.output_nodes,
            **self.kwargs,
        )
        return Data.DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=pad_seq_collate,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            persistent_workers=True,
            timeout=self.kwargs.get('timeout', 300),
        )

    def sp_ratio_wz(self):
        """Split into [train_loaders, val_loaders, test_loaders],
        each a list of length world_size."""
        tra, val, tst = _train_split(self.h5s, self.split_ratio)
        loaders = []
        for subset in [tra, val, tst]:
            parts = _evenly_split(subset, self.world_size)
            loaders.append([self._make_dl(p) for p in parts])
        return loaders

    def sp_wz(self):
        """Split all h5s across world_size for inference."""
        parts = _evenly_split(self.h5s, self.world_size)
        return [self._make_dl(p) for p in parts]
