# -*- coding: utf-8 -*-
import math
import os
import pathlib
import random
from typing import List

import h5py
import numpy as np
import scipy
import torch
import torch.utils.data as Data
from private_modules import load_yaml_config
from private_modules.Torch import MCFDS
from scipy import interpolate
from .utils import read_mat_file


# class WESTDS(Data.IterableDataset):
#     def __init__(self,
#                  files,
#                  input_nodes,
#                  output_nodes,
#                  dtype=np.float32,
#                  **kwargs) -> None:
#         super().__init__()
#         self.files = files
#         self.dtype = dtype
#         self.input_nodes = input_nodes
#         self.output_nodes = output_nodes
#         nodes = []
#         nodes.extend(self.input_nodes)
#         nodes.extend(self.output_nodes)
#         self.nodes = nodes
#         self.file_len = len(files)
#         self.kwargs = kwargs

#     def __len__(self,):
#         return self.file_len
    
#     def get_mat_file(self, mat_file):

#         data_dict = read_mat_file(mat_file, self.nodes)
        
#         node_data_list = []
#         node_flags = []
#         time_axis = data_dict['time']
#         for node in self.nodes:
#             node_data = data_dict[node]
#             if node_data is None:
#                 node_data = np.zeros_like(time_axis, dtype=self.dtype)
#                 node_flags.append(0)
#             else:
#                 inf_value = 3.2e32
#                 node_data = np.nan_to_num(
#                     node_data,
#                     posinf=inf_value,
#                     neginf=-inf_value,)
#                 node_flags.append(1)
#             node_data = np.array(node_data)[:, np.newaxis]
#             node_data_list.append(node_data)
#         nodes_data = np.concatenate(
#             node_data_list, 
#             dtype=self.dtype, axis=1)
#         return nodes_data, len(time_axis), node_flags
    
#     def gen_mat_data(self, iter_mat_files):
#         for mat_file in iter_mat_files:
#             data = self.get_mat_file(mat_file)
#             data = self.callback(data)
#             yield data
    
#     def __iter__(self):
#         # This is the distribute process key.
#         worker_info = Data.get_worker_info()
#         if worker_info is None:
#             iter_mats = self.files
#         else:
#             per_worker = int(
#                 math.ceil(self.file_len / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = per_worker * worker_id
#             iter_end = min(iter_start+per_worker, len(self.files))
#             iter_mats = self.files[iter_start:iter_end]
#         return self.gen_mat_data(iter_mats)

#     def callback(self, data):
#         MS_df = self.kwargs['MS_df']
#         means = MS_df.loc['mean', self.nodes].to_numpy()
#         stds = MS_df.loc['stDev', self.nodes].to_numpy()
#         XY = (data[0] - means) / stds
#         X = XY[..., :-len(self.output_nodes)]
#         Y = XY[..., -len(self.output_nodes):]
#         Y_len = data[1]
#         Y_flags = data[2][-len(self.output_nodes):]
#         info = data[-1]
#         return X, Y, Y_len, Y_flags, info


class StdWESTShotDS(MCFDS.__BaseH5__):
    def get_h5_data(self, h5_file):
        data_dict = read_mat_file(h5_file, self.nodes)
        node_data_list = []
        shot_node_flags = []
        info = dict({})
        time_axis = data_dict['time']
        for node in self.nodes:
            node_data = data_dict[node]
            if node_data is None:
                node_data = np.zeros_like(time_axis, dtype=self.dtype)
                shot_node_flags.append(0)
            else:
                inf_value = 3.2e32
                node_data = np.nan_to_num(
                    node_data,
                    posinf=inf_value,
                    neginf=-inf_value,)
                shot_node_flags.append(1)
            node_data = np.array(node_data)[:, np.newaxis]
            node_data_list.append(node_data)
        shot_data = np.concatenate(
            node_data_list, 
            dtype=self.dtype, axis=1)
        shot_len = shot_data.shape[0]
        info['file_name'] = h5_file
        return shot_data, shot_len, shot_node_flags, info

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
            timeout=300,
        )
        return dl
