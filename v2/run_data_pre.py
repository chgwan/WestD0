# -*- coding: utf-8 -*-
# data preparation
import pathlib
import os
from tqdm import tqdm
import h5py
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import data_gen, utils
from src.data_utils import (strongest_granger_causality, 
                            strongest_z_score, 
                            merge_mat_h5, filter_h5)
from private_modules import strpath2path
from private_modules import save_to_file, parse_args, MpRun
from private_modules.FedData import stat_node
import math

figs_dir = pathlib.Path('./Database/ConFigs/')
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
            #TODO: choose a different way to deal with Pha1-2 and LHW2
            p_matrix[input_idx, :] = np.nan
            continue
        for output_idx, sig_output in enumerate(output_arr.T):
            unique_values = np.unique(sig_output.round(decimals=4))
            # output did not have any missing values.
            if node_flags[input_dim+output_idx] == 0 or len(unique_values) < 10:
                p_matrix[:, output_idx] = np.nan
                continue
            if option == "gp":
                strongest_p = strongest_granger_causality(sig_output, sig_input, maxlag=maxlag)   
            elif option == "zp":
                # strongest_p = calc_corrcoef(sig_input, sig_output)
                strongest_p = strongest_z_score(sig_output, sig_input, maxlag)
            p_matrix[input_idx, output_idx] = strongest_p
    return p_matrix

def h5_p_matrix(h5_file, option='gp', maxlag=4):
    filter_wz = 99
    filter_func = data_gen.SMA
    input_nodes, output_nodes = utils.get_nodes()
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
    p_matrix = calc_gp(input_arr, output_arr, node_flags, option=option, maxlag=maxlag)
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

def calc_mean_p_matrix(p_matrix_f):
    p_matrix_list = np.load(p_matrix_f)
    nrows = p_matrix_list.shape[1]
    ncols = p_matrix_list.shape[2]
    p_matrix = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols): # only calc non-nan
            p_matrix[i, j] = np.mean(np.abs(p_matrix_list[:, i, j][~np.isnan(p_matrix_list[:, i, j])]))
    return p_matrix

def plt_p_matrix(option='gp'):
    plt.style.use('seaborn-v0_8-poster')
    stat_dir = 'Database/Stat'
    stat_dir = strpath2path(stat_dir)
    zp_matrix_f = stat_dir.joinpath('p_matrix_zp.npy')
    gp_matrix_f = stat_dir.joinpath('p_matrix_gp.npy')
    if option == "gp":
        gp_matrix = calc_mean_p_matrix(gp_matrix_f)
        data = gp_matrix
        title = "(a) Granger causality"
        fig_name = "granger_causality"
    elif option == "zp":
        zp_matrix = calc_mean_p_matrix(zp_matrix_f)
        data = zp_matrix
        title = "(b) Absolute correlation coefficient"
        fig_name = "correlation_coefficient"   
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    fig = plt.figure(figsize=(10, 10), dpi=200)
    # input_nodes, output_nodes = get_nodes()
    input_nodes, output_nodes = utils.get_render_nodes()
    sns.heatmap(data, annot=True, fmt=".2e", cmap=cmap,
                xticklabels=output_nodes, yticklabels=input_nodes[:],
                # vmin=0, vmax=0.15,
                linewidths=0.5, linecolor='white',
                cbar_kws={"label": "Score"},
                annot_kws={"size": 'large'})
    plt.title(f"{title} between input and output signals", fontsize=14)
    plt.tight_layout()
    fig_path = figs_dir.joinpath(f"{fig_name}.pdf")
    fig.savefig(fig_path)
    return fig

def plt_nodes_comparsion():
    shot = 57869
    input_nodes, output_nodes = utils.get_nodes()
    plot_nodes = ['Vloop_scope_3']
    plot_nodes.extend(output_nodes)
    data_dir = "$HOME/DATABASE/DataBase/WEST/WestData"
    data_dir = strpath2path(data_dir)
    h5_file = data_dir.joinpath(f"{shot}.h5")
    collect_data = []
    with h5py.File(h5_file, 'r') as hf:
        for node in plot_nodes:
            node_data = hf[node][()]
            collect_data.append(node_data)
    plt.close('all')
    n_cols = 3
    n_rows = math.ceil(len(plot_nodes) / n_cols)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, 
        figsize=(6.4 * n_cols, 4.8 * n_rows), 
        )

    axes = np.atleast_1d(axes).flatten()
    for i, (ax, sub_data) in enumerate(zip(axes, collect_data)):
        X = np.arange(len(sub_data)) / 1000
        ax.plot(X[2000:4000], sub_data[2000:4000])
        ax.set_title(plot_nodes[i])
    fig.suptitle(fr'WEST discharge #{shot}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figs_dir.joinpath(f'{shot}-output_comparsion.pdf'))

if __name__ == "__main__":
    args = parse_args()
    run_code = args.run_code.strip()
    run_code = f"{run_code}"
    eval(run_code)    