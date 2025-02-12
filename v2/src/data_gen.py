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

class StdWESTShotDS(MCFDS.MCFShotDataSet):
    def __init__(self, 
                 h5_list, 
                 input_nodes, 
                 output_nodes, 
                 *, 
                 dtype = np.float32, 
                 **kwargs):
        super().__init__(h5_list, input_nodes, output_nodes, dtype=dtype, **kwargs)
    
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
