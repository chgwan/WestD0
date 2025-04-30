# -*- coding: utf-8 -*-
import pathlib
import os
from src.data_utils import merge_mat_h5, filter_h5
from private_modules.utilities import save_to_file, parse_args, MpRun
from private_modules.FedData import stat_node
from tqdm import tqdm
import h5py
import shutil
import numpy as np
from src import data_gen
from src.data_utils import strongest_granger_causality
from private_modules import load_yaml_config
from private_modules.utilities import calc_corrcoef

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

def save_merge():
    def warp_merge(shot):
        h5_file = IMASH5_dir.joinpath(f'{shot}.h5')
        mat_file = Mat_dir.joinpath(f'DCS_archive_{shot}.mat')
        data_dict = merge_mat_h5(h5_file, mat_file)
        west_f = WestData_dir.joinpath(f'{shot}.h5')
        save_to_file(west_f, data_dict, True)

    IMASH5_dir = os.path.expandvars("$DATABASE_PATH/DataBase/WEST/IMASH5")
    IMASH5_dir = pathlib.Path(IMASH5_dir)
    Mat_dir = IMASH5_dir.parent.joinpath("PCS")
    WestData_dir = IMASH5_dir.parent.joinpath("WestData")

    shots = [int(shot.parts[-1][:-3]) for shot in IMASH5_dir.iterdir()]
    mp_run = MpRun(num_workers=10)
    mp_run.mp_no_return_run_func(warp_merge, shots)

def stat_nodes_h5():
    h5s_dir = os.path.expandvars('$DATABASE_PATH/DataBase/WEST/WestData')
    h5s_dir = pathlib.Path(h5s_dir)
    h5_stat_csv = pathlib.Path("./Database/Stat/h5_stat.csv")

    tmp_h5_file = next(h5s_dir.glob("*.h5"))
    with h5py.File(tmp_h5_file, 'r') as hf:
        all_keys = list(hf.keys())
    keys = []
    for key in all_keys:
        if not "time" in key:
            keys.append(key)
    calc_shots = [int(shot.parts[-1][:-3]) for shot in h5s_dir.glob("*.h5")]

    stat_node.stat(
        keys, 
        h5s_dir, 
        h5_stat_csv, 
        num_workers=40,
        shots=calc_shots,
        )

    MS_file = h5_stat_csv.parent.joinpath("h5_ms.csv")
    stat_node.calMS(keys, h5_stat_csv, MS_file)

def clear_h5():
    def warp_clean(h5):
        if filter_h5(h5):
            shutil.move(h5, Error_dir)
    h5s_dir = os.path.expandvars('$DATABASE_PATH/DataBase/WEST/WestData')
    h5s_dir = pathlib.Path(h5s_dir)
    Error_dir = os.path.expandvars('$DATABASE_PATH/DataBase/WEST/ErrorShots')
    h5s =  list(h5s_dir.glob("*.h5"))
    mp_run = MpRun(num_workers=10)
    mp_run.mp_no_return_run_func(warp_clean, h5s)    


def calc_gp(input_arr, output_arr, node_flags, option="gp", maxlag=4,):
    input_dim = input_arr.shape[1] # in our case is 19
    output_dim = output_arr.shape[1] # in our case is 6
    p_matrix = np.zeros((input_dim, output_dim))
    for input_idx, sig_input in enumerate(input_arr.T):
        unique_values = np.unique(sig_input.round(decimals=4))
        if node_flags[input_idx] == 0 or len(unique_values) == 1:
            p_matrix[input_idx, :] = np.nan
            continue
        for output_idx, sig_output in enumerate(output_arr.T):
            unique_values = np.unique(sig_output.round(decimals=4))
            if node_flags[input_dim+output_idx] == 0 or len(unique_values) == 1:
                p_matrix[:, output_idx] = np.nan
                continue
            if option == "gp":
                min_p = strongest_granger_causality(sig_output, sig_input, maxlag=maxlag)   
            elif option == "zp":
                min_p = calc_corrcoef(sig_input, sig_output)
            p_matrix[input_idx, output_idx] = min_p
    return p_matrix

def h5_p_matrix(h5_file, option='gp', maxlag=4):
    filter_wz = 99
    filter_func = data_gen.SMA
    input_nodes, output_nodes = get_nodes()
    nodes = []
    nodes.extend(input_nodes)
    nodes.extend(output_nodes)
    data, node_flags, timeAxis = data_gen.read_h5_tokamak(
        h5_file, 
        nodes, 
        filter_func=filter_func, 
        filter_wz=filter_wz)
    input_dim = len(input_nodes)
    input_arr = data[:, :input_dim]
    output_arr = data[:, input_dim:]
    p_matrix = calc_gp(input_arr, output_arr, node_flags, option=option)
    return p_matrix

def mp_h5_p_matrix(num_workers=64, option='gp', maxlag=4):
    """  mp running the p_matrix calculation

    Args:
        num_workers (int): as name
        option (Optional): `gp` or `zp`, 
            `gp` means strongest granger causality p-value
            `zp` means correlation coefficient p-value

    """
    h5s_dir = os.path.expandvars('$DATABASE_PATH/DataBase/WEST/WestData')
    h5s_dir = pathlib.Path(h5s_dir)
    h5s = list(h5s_dir.glob("*.h5"))

    # h5s = h5s[:4]
    my_mp_run = MpRun(num_workers)
    p_matrix_list = my_mp_run.mp_return_list_run_func(h5_p_matrix, 
                                                      h5s, 
                                                      option, 
                                                      maxlag)
    p_arr = np.stack(p_matrix_list)
    # p_arr = np.mean(p_arr, axis=0)
    stat_dir = "Database/Stat"
    p_arr_f = f"p_matrix_{option}.npy"
    p_arr_f = os.path.join(stat_dir, p_arr_f)
    np.save(p_arr_f, p_arr)


if __name__ == "__main__":
    args = parse_args()
    run_code = args.run_code.strip()
    run_code = f"{run_code}"
    eval(run_code)    