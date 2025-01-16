# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import pathlib
import os
# from private_modules import load_yaml_config
import yaml 
import pandas as pd

def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        Dict. The configuration information in dict format.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``
    """
    # Read YAML experiment definition file
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

mat_data_dir = '/donnees/NTU/NTU'

def read_mat_file(mat_file: os.PathLike, nodes):
    data = loadmat(mat_file)
    node_stat_dict = dict({})
    for scope_name in nodes:
        scope_struct = data[scope_name][0,0]
        time_field = scope_struct['time'].squeeze()
        # Extract signals and values
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0,0] 
        num_signals = signal_values.shape[1]
        for i in [0, 3]:
            if num_signals != 1:
                sig_data = signal_values[:, i]
                node_stat_dict[f"{scope_name}_{i}"] = True
                node_stat_dict[f"{scope_name}_{i}_sum"] = np.sum(sig_data)
                node_stat_dict[f"{scope_name}_{i}_square_sum"] = np.sum(sig_data**2,
                                                        dtype=np.float128)
                node_stat_dict[f"{scope_name}_{i}_mean"] = np.mean(sig_data)

            else:
                node_stat_dict[f"{scope_name}_{i}"] = False
                node_stat_dict[f"{scope_name}_{i}_sum"] = None
                node_stat_dict[f"{scope_name}_{i}_square_sum"] = None
                node_stat_dict[f"{scope_name}_{i}_mean"] = None
            node_stat_dict[f"{scope_name}_{i}_length"] = len(time_field)    
    node_stat_dict["length"] = len(time_field)          
    return node_stat_dict

def scan_dir(file_list, nodes, stat_file):
    data_list = []
    shots = []
    for file in file_list:
        shot = file.name[:-4].split('_')[-1]
        shots.append(int(shot))
        rtn_dict = read_mat_file(file, nodes)
        data_list.append(rtn_dict)
    df = pd.DataFrame.from_records(data_list, index=shots)
    df.to_csv(stat_file)

def calMS(nodes, stat_file: pathlib.Path, MS_file: pathlib.Path, **kwargs):
    """ Mapreduce formula see https://zhuanlan.zhihu.com/p/48025855
    .. math::
        $$
        \begin{aligned}
        s^2 & =\frac{1}{N-1} \sum_{i=0}^{N-1}\left(X_i-\bar{X}\right)^2 \\
        & =\frac{1}{N-1}\left[\sum X_i^2-\frac{1}{N}\left(\sum X_i\right)^2\right]
        \end{aligned}
        $$
    .. math::
    """
    stat_df = pd.read_csv(stat_file, index_col=0, low_memory=False)
    MS_df = pd.DataFrame(index=["mean", "stDev"], columns=nodes)
    for node in nodes:
        # determined by square sum.
        finit_index = np.isfinite(stat_df.loc[:, "%s_square_sum" % node].to_numpy(dtype=np.float128))
        existence_index = stat_df.loc[:, node]
        index = finit_index & existence_index
        index_row = stat_df.index[index].to_numpy()
        # temp_df = stat_df.loc[index_row, :]
        # x = temp_df.loc[:,"%s_square_sum"%node]
        # remove large data
        # larger_shots = x.sort_values()[-400:].index
        # index_row = np.setdiff1d(index_row, larger_shots)
        calc_df = stat_df.loc[index_row, :]
        N_sample = np.sum(calc_df.loc[:, "length"])
        if node in kwargs.get('d2_signal_list', []):
            idx = kwargs['d2_signal_list'].index(node) 
            num_channels = kwargs['num_channel_list'][idx]
            N_sample = N_sample * num_channels
        sample_sum = np.sum(calc_df.loc[:, "%s_sum" % node].to_numpy(dtype=np.float128))
        sample_square_sum = np.sum(calc_df.loc[:, "%s_square_sum" % node].to_numpy(dtype=np.float128))
        mean = sample_sum / N_sample
        stDev = np.sqrt(1/(N_sample)*(sample_square_sum -
                        1/N_sample*(sample_sum**2)))
        MS_df.loc["mean", node] = mean
        MS_df.loc["stDev", node] = stDev
    MS_df.to_csv(MS_file)
    
if __name__ == "__main__":
    mat_data_dir = '/donnees/NTU/NTU'
    database_dir = '/home/CW037006/Code/DataTest/Database'
    database_dir = pathlib.Path(database_dir)
    mat_data_dir = pathlib.Path(mat_data_dir)
    file_list = list(mat_data_dir.glob("DCS_archive_*.mat"))
    file_list = file_list[:]
    config_path = '/home/CW037006/Code/DataTest/config/data_config.yml'
    config = load_yaml_config(config_path)
    nodes = config["nodes"]
    df = scan_dir(file_list, nodes)
    df.to_csv(database_dir.joinpath("Stat/node_stat.csv"))
    # file = '/donnees/NTU/NTU/DCS_archive_57604.mat'
    # we have 677 shots. 
    # print(len(file_list))





