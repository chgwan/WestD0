# -*- coding: utf-8 -*-
# --------------------------------------------------------
# A generalized, tools for easy create python script.
# Main for data preprocessing or postprocessing.
# Copyright (c) 2025 Chenguang Wan
# Licensed under the attached License [see LICENSE for details]
# Writen by Chenguang Wan on 2025-02-11
# --------------------------------------------------------

from private_modules.Torch import MCFDS
import numpy as np
from torch.utils import data as Data
import h5py
import os
from typing import LiteralString, List

def findTFEnd(Ip_ref, time):
    # Ip_ref in 6 digits.
    IpMax = max(Ip_ref)
    third_Ip = IpMax // 3
    ids = Ip_ref > third_Ip
    d_Ip_ids = np.abs(np.gradient(Ip_ref, time)) < 500
    TFEnd = time[d_Ip_ids & ids][-1]
    return TFEnd

def read_h5_tokamak(
        h5_file:os.PathLike, 
        nodes:List[LiteralString],
        dtype=np.float32,
        **kwargs):
    filter_func = kwargs['filter_func']
    filter_wz = kwargs['filter_wz'] # filter window size
    half_filter_wz = filter_wz // 2
    with h5py.File(h5_file, mode="r") as hf:
        timeAxis = hf["time"][()]
        Ip_ref = hf['Ip_scope_0'][()]
        TFEnd = findTFEnd(Ip_ref, timeAxis)
    
        # only select the IMAS time scope.
        IMAS_time = hf['IMAS_time'][()]
        IMAS_start, IMAS_end = IMAS_time[0], IMAS_time[-1]
        ids0 = timeAxis >= IMAS_start
        # reduce the training using in ramp-down phase. 
        ids1 = timeAxis <= min((IMAS_end - TFEnd) * 0.8 + TFEnd, IMAS_end)
        # ids1 = timeAxis <= IMAS_end
        ids = ids0 & ids1

        timeAxis = timeAxis[ids]
        timeAxis = timeAxis[half_filter_wz: - half_filter_wz]
        # timeLen = len(timeAxis)
        # data:np.ndarray = np.empty((timeLen, 0), dtype=dtype)
        data = []
        node_flags = []
        for node in nodes:
            # col = db[node]
            # nodeDict = col.find_one({"shot":shot})
            # `Node` means inexistence.
            if hf[node].shape is None or len(np.unique(hf[node][()])) == 1:
                nodeData = np.zeros_like(timeAxis, dtype=dtype)
                node_flags.append(0)
            else:
                node_flags.append(1)
                nodeData = hf[node][()][ids]
                inf_value = 3.2e32
                nodeData = np.nan_to_num(
                    nodeData,
                    posinf=inf_value,
                    neginf=-inf_value,)
                nodeData = filter_func(nodeData, filter_wz)
                # if the times length is unsame with nodeData length
                assert len(timeAxis) == len(nodeData), \
                    "file: %s, Node:%s" % (h5_file, node)
                # nodeData = np.interp(timeAxis, times, nodeData)
            if len(nodeData.shape) == 1:
                nodeData = np.array(nodeData)[:, np.newaxis]
            # data = np.append(data, nodeData, axis=1)
            data.append(nodeData)
        node_flags = np.array(node_flags, dtype=int)
        # data = np.stack(data, dtype=dtype, axis=1)
        data = np.concatenate(data, dtype=dtype, axis=1)
    return data, node_flags, timeAxis

def SMA(data, window_size):
    if window_size <= 0:
        raise ValueError("Window size of SMA must be greater than zero.")    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

class StdWESTShotDS(MCFDS.MCFShotDataSet):
    def __init__(self, 
                 h5_list, 
                 input_nodes, 
                 output_nodes, 
                 *, 
                 read_h5 = read_h5_tokamak, 
                 dtype = np.float32, 
                 **kwargs):
        super().__init__(h5_list, input_nodes, output_nodes, read_h5=read_h5, dtype=dtype, **kwargs)
    
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

class StdWestShotWinDS(MCFDS.MCFShotWinDataSet):
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

    
class H5GenDataLoader(MCFDS.H5GenDL):
    def get_DL(self, iter_h5s = None):
        if 'window_size' in self.kwargs:
            stat_df = self.kwargs['stat_df']
            shots = []
            if iter_h5s == None: iter_h5s = self.h5s
            for h5 in iter_h5s:
                # pathlib.Path(h5)
                _, tail = os.path.split(h5)
                shots.append(int(tail[:-3]))
            lengths = stat_df.loc[shots, 'length']
            total_length = sum(lengths)      
            ds = self.DS(
                iter_h5s,
                total_length=total_length,
                **self.kwargs,
                )
        else:
            ds = self.DS(
                iter_h5s,
                **self.kwargs,
                )
        dl = Data.DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=MCFDS.pad_seq_collate,
            pin_memory=self.kwargs.get(
                'pin_memory', False),
            prefetch_factor=2,
            persistent_workers=True,
            timeout=self.kwargs.get('timeout', 300),
        )
        return dl


if __name__ == "__main__":
    pass
