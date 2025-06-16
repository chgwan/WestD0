# -*- coding: utf-8 -*-
# some funcs are used to conclusion plot
import math
import os
import pathlib

import h5py
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

from . import data_gen
from .utils import get_nodes
from private_modules.utilities import convert_hdf5_2dict
plt.style.use('seaborn-v0_8-paper')
plt.ioff()

default_config_f = "$HOME/Papers/WestD0/v2/configs/former.yml"
default_config_f = os.path.expandvars(default_config_f)


def calc_r2_nodes(pred_h5, org_h5):
    """  We have two h5 files, predicted and orignal h5 files
    """
    with h5py.File(org_h5, 'r') as hf:
        Ip_ref = hf['Ip_scope_0'][()]
        timeAxis = hf['time'][()]
        TF_start, TF_end = data_gen.findTF(Ip_ref, timeAxis)

    with h5py.File(pred_h5, 'r') as hf:
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf['Y_tgt'][()]

    timeLen_org = len(timeAxis)
    timeLen_pred = Y_hat.shape[0]
    half_time_len = (timeLen_org - timeLen_pred) // 2

    dummy_ids = (TF_start < timeAxis) & (timeAxis <= TF_end)
    select_ids = np.arange(timeAxis.shape[0])[dummy_ids]

    TF_start_idx, TF_end_idx = \
        min(select_ids) - half_time_len, max(select_ids) - half_time_len
    
    # missing one value
    Y_hat_avg = np.average(Y_hat[TF_start_idx:TF_end_idx])
    Y_tgt_avg = np.average(Y_tgt[TF_start_idx:TF_end_idx])
    return Y_hat_avg, Y_tgt_avg

def calc_metrics(h5_file, metric_names, strip_length=0, MS_f=None, config_f=None, ):
    if MS_f is None:
        MS_f = '$HOME/Papers/WestD0/v2/Database/Stat/h5_global_MS.csv'
        MS_f = os.path.expanduser(MS_f)
    MS_df = pd.read_csv(MS_f, index_col=0)
    input_nodes, output_nodes = get_nodes(config_f)   
    output_mean_stds = MS_df.loc[:, output_nodes].to_numpy()  
    metrics = dict({})
    with h5py.File(h5_file) as hf:
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf['Y_tgt'][()]
        if strip_length != 0:
            Y_hat = Y_hat[strip_length:-strip_length]
            Y_tgt = Y_tgt[strip_length:-strip_length]
    
    Y_hat = (Y_hat - output_mean_stds[0, :]) / output_mean_stds[1, :]
    Y_tgt = (Y_tgt - output_mean_stds[0, :]) / output_mean_stds[1, :]
    # calc metrics
    if 'r2' in metric_names:
        r2 = r2_score(Y_tgt, Y_hat)
        metrics['r2'] = r2
    if 'mse' in metric_names:
        mse = np.mean((Y_hat - Y_tgt) ** 2)
        metrics['mse'] = mse
    metrics['file'] = str(h5_file)
    return metrics


def scatter_list(r2s_list, plt_func, **kwargs):
    plt.close('all')
    arr_len = len(r2s_list)
    n_rows = arr_len // 2 + 1
    # fig = plt.figure(constrained_layout=False, figsize=(6.8 * n_rows, 4.8 * 2), dpi=300)
    fig_kwargs = kwargs['fig_kwargs']
    fig = plt.figure(**fig_kwargs)
    gs = gridspec.GridSpec(n_rows, 2)
    n_rows = gs.nrows
    n_cols = gs.ncols

    ylabel = kwargs['ylabel']
    signal_names = kwargs['signal_names']
    for ix in range(len(r2s_list)):
        signal_name = signal_names[ix]
        r2s = r2s_list[ix]
        row = ix // n_cols
        col = ix % n_cols
        sub_gs = gs[row, col]
        plt_func(r2s, sub_gs, ylabel, signal_name, fig)
        # sub = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[row, col], wspace=0)
        # ax0 = fig.add_subplot(sub[:, :-1])
        # ax1 = fig.add_subplot(sub[:, -1])
    return fig

def scatter_r2s(r2s, 
                sub_gs:gridspec.GridSpec, 
                ylabel:str, 
                signal_name:str, 
                fig):
    """  
    """
    sub = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=sub_gs, wspace=0)
    ax0 = fig.add_subplot(sub[:, :-1])
    ax1 = fig.add_subplot(sub[:, -1])
    ax1.tick_params(labelbottom=False, labelleft=False)
    test_num = r2s.shape[0]
    scatter_plt = ax0.scatter(x=np.arange(test_num),
                            y=r2s,
                            c="tab:blue",
                            alpha=0.6)
    ax0.set_xticks([])
    ax0.set_xlabel("Simulation id")
    ax0.set_ylabel(fr"{ylabel}")
    ax0.set_ylim(0, 1)

    ax1.set_ylim(0, 1)
    sns.histplot(y=r2s, color="b", kde=True, ax=ax1, linewidth=0)
    ax1.set_xlabel("")
    avg_sim = np.mean(r2s)
    ax0.set_title(fr"{signal_name}'s average {ylabel} is {avg_sim:.3f}")
    return fig


def plt_pred_h5(
        h5_file, 
        nodes_name=[r'$\beta_{n}$', r'$\beta_{p}$', 
                    r'$\beta_{t}$', r'$w_{mhd}$', 
                    r'$q_0$', r'$q_{95}$'],
        src_h5 = None,
        heating_nodes = [],
        filter_wz = None, 
        fkwargs = dict({}),):
    with h5py.File(h5_file, 'r') as hf:
        keys = list(hf.keys())
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf["Y_tgt"][()]
    shot = int(os.path.basename(h5_file)[:-3])

    plt.close('all')
    n_cols = 2
    n_rows = math.ceil((len(nodes_name) + len(heating_nodes)) / n_cols)
    lw = 1.5
    alpha = 1
    markers = ['D', '*', 'X']
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=[6.8 * n_cols, 4.8 * n_rows], 
                             sharex=True,
                             **fkwargs,)
    X = np.arange(Y_hat.shape[0]) / 1000
    if len(heating_nodes) != 0:
        heating_data = get_truncate_src_data(src_h5, heating_nodes, filter_wz=filter_wz)
        for node_idx, node_name in enumerate(heating_nodes):
            row_idx = node_idx // n_cols
            col_idx = node_idx % n_cols
            ax = axes[row_idx][col_idx]
            ax.plot(X, heating_data[:, node_idx],
                c='b', 
                lw=lw, 
                alpha=alpha, 
                label=node_name)

    # for node_idx, node_name
    for node_idx, name in enumerate(nodes_name):
        row_idx = (node_idx + len(heating_nodes)) // n_cols
        col_idx = (node_idx + len(heating_nodes)) % n_cols
        ax = axes[row_idx][col_idx]
        ax.plot(
            X, 
            Y_tgt[:, node_idx], 
            c = 'b', 
            lw=lw,
            alpha=alpha,
            label=fr'Target of {name}'
        )
        ax.plot(
            X, 
            Y_hat[:, node_idx], 
            c = 'r', 
            # marker=markers[0],
            # markevery=10,
            lw=lw,
            alpha=alpha,
            label=fr'Prediction of {name}')
        # if node_idx == 3:
        ax.set_ylim(ymax=max(Y_tgt[:, node_idx]) * 1.15)
        ax.legend(loc='upper right')
    fig.suptitle(f'West shot: {shot}')
    # axes[0].set_title(f'West shot: {shot}')
    for i in range(n_cols):
        axes[-1][i].set_xlabel('Time [s]')
    # plt.tight_layout()
    return fig

def plt_h5(h5_file, nodes, ncols = 4):
    nodes_num = len(nodes)
    nrows = math.ceil(nodes_num / ncols)
    fig_kwargs = {"constrained_layout": True,
                "figsize": (6.8 * ncols, 4.8 * nrows),
                "dpi": 300}
    lw = 1.5
    alpha = 1
    plt.close('all')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **fig_kwargs)
    with h5py.File(h5_file, 'r') as hf:
        time = hf['time'][()]
        for node_idx, node in enumerate(nodes):
            node_data = hf[node][()]
            row_idx = node_idx // ncols
            col_idx = node_idx % ncols
            ax = axes[row_idx][col_idx]
            ax.plot(time, 
                    node_data, 
                    # c='r',
                    lw=lw, alpha=alpha,
                    label=fr'{node}')
            ax.legend(loc='upper right')
    shot = os.path.basename(h5_file)
    shot = shot[:-3]
    for i in range(ncols):
        axes[-1][i].set_xlabel('Time [s]')
    fig.suptitle(f'Shot: {shot}')
    # plt.tight_layout()
    return fig

def get_truncate_src_data(src_h5, nodes, filter_wz, dtype=np.float64):
    with h5py.File(src_h5, mode="r") as hf:
        timeAxis = hf["time"][()]

        # only select the IMAS time scope.
        IMAS_time = hf['IMAS_time'][()]
        IMAS_start, IMAS_end = IMAS_time[0], IMAS_time[-1]
        time_start = IMAS_start
        time_end = IMAS_end
        ids0 = timeAxis >= time_start
        ids1 = timeAxis < time_end
        ids = ids0 & ids1   

        half_filter_wz = filter_wz // 2
        timeAxis = timeAxis[ids]
        timeAxis = timeAxis[half_filter_wz: - half_filter_wz]
        data = []
        for node in nodes:
            # col = db[node]
            # nodeDict = col.find_one({"shot":shot})
            # `Node` means inexistence.
            if hf[node].shape is None:
                nodeData = np.zeros_like(timeAxis, dtype=dtype)
            else:
                nodeData = hf[node][()][ids]
                nodeData = nodeData[half_filter_wz:-half_filter_wz]
            if len(nodeData.shape) == 1:
                nodeData = np.array(nodeData)[:, np.newaxis]
            data.append(nodeData)
        data = np.concatenate(data, dtype=dtype, axis=1)
        return data


def plt_pred_h5_vertically(
    h5_file, 
    nodes_name=[r'$\beta_{n}$', r'$\beta_{p}$', 
            r'$\beta_{t}$', r'$W_{mhd}$', 
            r'$q_0$', r'$q_{95}$'],
    *, 
    src_h5 = None, 
    hs_in_use = ['LHW'],
    filter_wz = None, 
    fkwargs = dict({}),
    sub_order = None, 
    ):
    shot = int(os.path.basename(h5_file)[:-3])
    node_map = dict({
        'Ohmic': ['PowLH1_scope_3'],
        'LHW': ["PowLH1_scope_3", "PowLH2_scope_3"],
        'ICRH': ["PowIC1_scope_3", "PowIC2_scope_3", "PowIC3_scope_3",],
    })
    name_unit_map = {'LHW': 'LHW [W]',
                     'ICRH': 'ICRH [W]',
                     r'$W_{mhd}$': r'$W_{mhd}$ [J]'}

    with h5py.File(h5_file, 'r') as hf:
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf["Y_tgt"][()]

    plt.close('all')
    lw = 1.5
    alpha = 1
    markers = ['D', '*', 'X']

    n_rows = len(nodes_name) + len(hs_in_use)
    n_cols = 1
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=[6.8 * n_cols, 4.8 * n_rows], 
                             sharex=True,
                             **fkwargs,)
    
    X = np.arange(Y_hat.shape[0]) / 1000
    colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, dummy_hs in enumerate(hs_in_use):
        nodes = node_map[dummy_hs]
        node_data = get_truncate_src_data(src_h5, nodes, filter_wz=filter_wz,)
        ax = axes[idx]
        for sig_node_data in node_data.T:
            ax.plot(X, sig_node_data, c=colors_cycle[idx])
        text = dummy_hs
        if dummy_hs == 'Ohmic':
            text = 'No auxiliary'
        ax.text(0.2, 0.85, f"{text} heating", transform=ax.transAxes,
                    ha='center', va='center')
        name_unit = name_unit_map.get(dummy_hs, None)
        ax.set_ylabel(name_unit)
    
    for node_idx, name in enumerate(nodes_name):
        row_idx = node_idx + len(hs_in_use)
        ax = axes[row_idx]
        ax.plot(
            X, 
            Y_tgt[:, node_idx], 
            # c = 'b', 
            lw=lw,
            alpha=alpha,
            label=fr'Target of {name}'
        )
        ax.plot(
            X, 
            Y_hat[:, node_idx], 
            # c = 'r-', 
            ls = '--',
            # marker=markers[0],
            # markevery=10,
            lw=lw,
            alpha=alpha,
            label=fr'Prediction of {name}')      
        name_unit = name_unit_map.get(name, None)
        ax.set_ylabel(name_unit)
        ax.set_ylim(ymax=max(Y_tgt[:, node_idx]) * 1.15)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{sub_order} West shot: {shot}')
    return fig
            
