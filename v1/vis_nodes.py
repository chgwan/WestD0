# -*- coding: utf-8 -*-
import pandas as pd
import pathlib
import numpy as np
# from scipy.io import 
import os
from src.ML import utils, data_gen
# from matplotlib import pyplot as plt
from private_modules import load_yaml_config
from private_modules.Torch import MCFDS
import torch.utils.data as Data
from matplotlib import pyplot as plt

Database_dir = '/home/chenguang.wan/Papers/DataTest/Database'
Database_dir = pathlib.Path(Database_dir)
mat_dir = os.path.expandvars("$DATABASE_PATH/DataBase/WEST/PCS")
mat_dir = pathlib.Path(mat_dir)

config_file = '/home/chenguang.wan/Papers/DataTest/Database/configs/data_config.yml'
config = load_yaml_config(config_file)

data_params = config['data']
node_maps = config['nodes']
input_nodes, output_nodes = [], []

input_list = data_params['input_list']
output_list = data_params['output_list']

for nodeList_name in input_list:
    if "_real" in nodeList_name:
        dummy_list = node_maps[nodeList_name[:-5]]
        dummy_list = [f"{node}_3" for node in dummy_list]
        input_nodes.extend(dummy_list)
    elif "_ref" in nodeList_name:
        dummy_list = node_maps[nodeList_name[:-4]]
        dummy_list = [f"{node}_0" for node in dummy_list]
        input_nodes.extend(dummy_list)

for nodeList_name in output_list:
    if "_real" in nodeList_name:
        dummy_list = node_maps[nodeList_name[:-5]]
        dummy_list = [f"{node}_3" for node in dummy_list]
        output_nodes.extend(dummy_list)
    elif "_ref" in nodeList_name:
        dummy_list = node_maps[nodeList_name[:-4]]
        dummy_list = [f"{node}_0" for node in dummy_list]
        output_nodes.extend(dummy_list)

nodes = []
nodes.extend(input_nodes)
nodes.extend(output_nodes)

def get_discharge_interval(h5_file):
    start_idx, end_idx = utils.get_ref_interval(h5_file)
    start_idx = start_idx + 100
    end_idx = start_idx + 10000
    return start_idx, end_idx

def plt_h5_file(h5_file, fig_dir):
    dtype = np.float32
    data_dict = utils.read_mat_file(h5_file, nodes)
    start_idx, end_idx = get_discharge_interval(h5_file)
    # the real start point.
    node_data_list = []
    shot_node_flags = []
    info = dict({})
    time_axis = data_dict['time']
    for node in nodes:
        node_data = data_dict[node]
        if node_data is None:
            node_data = np.zeros_like(time_axis, dtype=dtype)
            shot_node_flags.append(0)
        else:
            inf_value = 3.2e32
            node_data = np.nan_to_num(
                node_data,
                posinf=inf_value,
                neginf=-inf_value,)
            shot_node_flags.append(1)
        node_data = np.array(node_data)
        cutted_node_data = node_data[start_idx:end_idx]
        cutted_time_axis = time_axis[start_idx:end_idx]
        node_data_list.append(node_data)
    
        plt.close('all')
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(time_axis, node_data)
        axes[0].axvspan(xmin=time_axis[start_idx], 
                        xmax=time_axis[end_idx],
                        color='lightgreen', 
                        alpha=0.2)
        axes[1].plot(cutted_time_axis, cutted_node_data)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Amplitude')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Amplitude')
        fig.suptitle(f'{node}')
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"{node}.png")
        fig.savefig(fig_path)

    # shot_data = np.concatenate(
    #     node_data_list, 
    #     dtype=dtype, axis=1)
    # shot_len = shot_data.shape[0]
    # info['file_name'] = h5_file
    # info['time_axis'] = time_axis[start_idx:end_idx]
    # return shot_data, shot_len, shot_node_flags, info

if __name__ == "__main__":
    # shot = 57604
    shot = 60282
    mat_file = f'$DATABASE_PATH/DataBase/WEST/PCS/DCS_archive_{shot}.mat'
    mat_file = os.path.expandvars(mat_file)

    fig_dir = '/home/chenguang.wan/Papers/DataTest/Database/compared_figs'
    plt_h5_file(mat_file, fig_dir)