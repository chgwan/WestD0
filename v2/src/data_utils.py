# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
from private_modules.utilities import convert_hdf5_2dict
import h5py
# # align data with the actual discharge start based on Ip_ref. 
# def read_mat_file(mat_file: os.PathLike, nodes):
#     data = loadmat(mat_file)
#     node_stat_dict = dict({})
#     # assume same time axis. 
#     Ip_struct = data['Ip_scope'][0,0]
#     time_field = Ip_struct['time'].squeeze()
#     node_stat_dict['time'] = time_field
#     for node in nodes:
#         scope_name = node[:-2]
#         slice = int(node[-1])
#         scope_struct = data[scope_name][0,0]
#         # time_field = scope_struct['time'].squeeze()
#         # Extract signals and values
#         signals_field = scope_struct['signals']
#         signal_values = signals_field['values'][0,0] 
#         # num_signals = signal_values.shape[1] 
#         signal_data = signal_values[:, slice]
#         node_stat_dict[node] = signal_data
#     return node_stat_dict

def read_all_scope(mat_file: os.PathLike):
    data = loadmat(mat_file)
    scope_keys = [key for key in data.keys() if key.endswith('_scope')]
    # print(scope_keys)
    scope_dict = dict({})
    scope_struct = data['Ip_scope'][0,0]
    time_field = scope_struct['time'].squeeze()
    Ip_scope_0 = scope_struct['signals']['values'][0, 0][:, 0]

    scope_dict['time'] = time_field
    ids = Ip_scope_0 > 2
    start_idx = min(np.arange(len(ids))[ids])
    start_time = time_field[start_idx]
    for scope_name in scope_keys:
        # Each scope is expected to be a struct at data[scope_name][0,0]
        scope_struct = data[scope_name][0,0]
        # Extract signals and values
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0,0]
        for i in [0, 3]:
            signal_value = signal_values[:, i]
            scope_dict[f'{scope_name}_{i}'] = signal_value
    return scope_dict, start_time


def merge_mat_h5(h5, mat_file):
    r""" merge
    Args:
        h5: the filepath of IMAS h5 file.
        mat_file: the filepath of DCS matfile.
    """
    mat_dict, start_time = read_all_scope(mat_file)
    # all discharge start exactly from 32
    dt = 32 - start_time
    # align the time to accurate discharge start
    mat_dict['time'] = mat_dict['time'] + dt

    sample_rate = 1e-3
    h5_dict = convert_hdf5_2dict(h5)

    end_time = mat_dict['time'][-1]
    time_axis = np.arange(32, end_time, sample_rate)
    time_axis_before = np.arange(mat_dict['time'][0], 32, sample_rate)

    h5_nodes = list(h5_dict.keys())
    h5_nodes.remove('time')
    mat_nodes = list(mat_dict.keys())
    mat_nodes.remove('time')

    data_dict = {}
    for node in mat_nodes:
        data_dict[node] = np.interp(time_axis, mat_dict['time'], mat_dict[node])
        data_dict[f"{node}_before"] = np.interp(time_axis_before, mat_dict['time'], mat_dict[node])
    for node in h5_nodes:
        data_dict[node] = np.interp(time_axis, h5_dict['time'], h5_dict[node])
    data_dict['time'] = time_axis
    data_dict['time_before'] = time_axis_before
    data_dict['PCS_time'] = mat_dict['time']
    data_dict['IMAS_time'] = h5_dict['time']
    return data_dict

def findTFEnd(Ip_ref, time):
    # Ip_ref in 6 digits.
    IpMax = max(Ip_ref)
    third_Ip = IpMax // 3
    ids = Ip_ref > third_Ip
    d_Ip_ids = np.abs(np.gradient(Ip_ref, time)) < 500
    TFEnd = time[d_Ip_ids & ids][-1]
    return TFEnd

def filter(h5):
    with h5py.File(h5, 'r') as hf:
        Ip_ref = hf['Ip_scope_0'][()]
        time = hf['time'][()]
        TFEnd = findTFEnd(Ip_ref, time)
        
        IMAS_time = hf['IMAS_time'][()]
        IMAS_start, IMAS_end = IMAS_time[0], IMAS_time[-1]
        if IMAS_end - TFEnd < 0:
            return False
        else:
            return True     


if __name__ == "__main__":
    read_all_scope("")