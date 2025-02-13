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

class StdWESTShotDS(MCFDS.MCFShotDataSet):
    def __init__(self, 
                 h5_list, 
                 input_nodes, 
                 output_nodes, 
                 *, 
                 dtype = np.float32, 
                 **kwargs):
        super().__init__(h5_list, input_nodes, output_nodes, dtype=dtype, **kwargs)
    
    def get_h5_data(self, h5_file):
        nodes = self.nodes
        dtype = self.dtype
        info = dict({})
        with h5py.File(h5_file, mode="r") as hf:
            times = hf["time"][()]
            timeAxis = hf["time"][()]
            timeLen = len(timeAxis)
            # data:np.ndarray = np.empty((timeLen, 0), dtype=dtype)
            data = []
            node_flags = []
            for node in nodes:
                # col = db[node]
                # nodeDict = col.find_one({"shot":shot})
                # `Node` means inexistence.
                if hf[node].shape is None or len(np.unique(hf[node][()])):
                    nodeData = np.zeros_like(timeAxis, dtype=dtype)
                    node_flags.append(0)
                else:
                    node_flags.append(1)
                    nodeData = hf[node][()]
                    inf_value = 3.2e32
                    nodeData = np.nan_to_num(
                        nodeData,
                        posinf=inf_value,
                        neginf=-inf_value,)
                    # if the times lengeh is unsame with nodeData length
                    assert len(times) == len(nodeData), \
                        "file: %s, Node:%s" % (h5_file, node)
                    # nodeData = np.interp(timeAxis, times, nodeData)
                if len(nodeData.shape) == 1:
                    nodeData = np.array(nodeData)[:, np.newaxis]
                # data = np.append(data, nodeData, axis=1)
                data.append(nodeData)
            node_flags = np.array(node_flags, dtype=int)
            # data = np.stack(data, dtype=dtype, axis=1)
            data = np.concatenate(data, dtype=dtype, axis=1)
        info['file_name'] = h5_file
        return data, data.shape[0], node_flags, info

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
    def get_DL(self, iter_h5s):
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
