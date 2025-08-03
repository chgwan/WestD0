# -*- coding: utf-8 -*-
# some funcs are used to conclusion plot
import math
import os
import pathlib

import h5py
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

from private_modules import convert_hdf5_2dict, load_yaml_config, strpath2path
from private_modules.utils.com_tools import calc_corrcoef
from private_modules import convert_hdf5_2dict
from . import data_gen
from .utils import get_nodes, get_render_nodes
from typing import *

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

plt.style.use('seaborn-v0_8-paper')
plt.ioff()

default_config_f = "$HOME/Papers/WestD0/v2/configs/former.yml"
default_config_f = strpath2path(default_config_f)
default_config = load_yaml_config(default_config_f)

default_ms_f = '$HOME/Papers/WestD0/v2/Database/Stat/h5_global_MS.csv'
default_ms_f = strpath2path(default_ms_f)



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

def __warp_r2_score__(tgt, hat, is_feature_wise):
    """
    Computes the coefficient of determination (R² score) between the target and prediction.

    Args:
        tgt (Tensor or array-like): Ground truth target values of shape (N,) or (N, F),
            where N is the number of samples and F is the number of features.
        hat (Tensor or array-like): Predicted values of the same shape as `tgt`.
        feature_wise (bool, optional): If True, computes R² for each feature independently.
            If False, computes a single aggregated R² score. Default: False.

    Returns:
        r2 (float or List[float]): A single R² score if `feature_wise=False`,
        otherwise a list of R² scores for each feature.
    """
    if is_feature_wise:
        r2s = r2_score(tgt, hat, multioutput='raw_values').tolist()
    else:
        r2s = r2_score(tgt, hat)
    return r2s

def __warp_mse_score__(
    tgt: Union[np.ndarray, list],
    hat: Union[np.ndarray, list],
    feature_wise: bool = False
) -> Union[float, list]:
    tgt = np.asarray(tgt)
    hat = np.asarray(hat)

    if feature_wise:
        return np.mean((hat - tgt) ** 2, axis=0).tolist()
    return float(np.mean((hat - tgt) ** 2))

def calc_metrics(
        h5_file, 
        metric_names, 
        *,
        is_feature_wise = False, # if false, will average all the feature to 1      
        strip_length=0, 
        MS_f=default_ms_f, 
        config_f=default_config_f, ):
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
    
    # convert to normalized data.
    Y_hat = (Y_hat - output_mean_stds[0, :]) / output_mean_stds[1, :]
    Y_tgt = (Y_tgt - output_mean_stds[0, :]) / output_mean_stds[1, :]
    
    n_features = Y_tgt.shape[1]
    # calc metrics
    if 'r2' in metric_names:
        r2 = __warp_r2_score__(Y_tgt, Y_hat, is_feature_wise)
        metrics['r2'] = r2
    if 'mse' in metric_names:
        mse = __warp_mse_score__(Y_tgt, Y_hat, is_feature_wise)
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
                sub_gs: gridspec.GridSpec, 
                ylabel: str, 
                ax_title: str, 
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
    # avg_sim = np.mean(r2s)
    # ax0.set_title(fr"{ax_title} Average {latex_name} is {avg_sim:.3f}")
    ax0.set_title(fr"{ax_title}")
    return fig

def plt_pred_h5(
        h5_file, 
        nodes_name=[r'$\beta_{n}$', r'$\beta_{p}$', 
                    r'$\beta_{t}$', r'$W_{mhd}$', 
                    r'$q_{95}$', r'$q_0$'],
        src_h5 = None,
        hs_in_use = [],
        filter_wz = None, 
        fkwargs = dict({}),):
    with h5py.File(h5_file, 'r') as hf:
        keys = list(hf.keys())
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf["Y_tgt"][()]
    shot = int(os.path.basename(h5_file)[:-3])

    plt.close('all')
    n_cols = 2
    n_rows = math.ceil((len(nodes_name) + len(hs_in_use)) / n_cols)
    lw = 1.5
    alpha = 1
    markers = ['D', '*', 'X']
    if "figsize" not in fkwargs.keys():
        figsize=[6.8 * n_cols, 4.8 * n_rows]
        fkwargs['figsize'] = figsize
    fig, axes = plt.subplots(n_rows, n_cols,  
                             sharex=True,
                             **fkwargs,)
    X = np.arange(Y_hat.shape[0]) / 1000
    if len(hs_in_use) != 0:
        heating_data = get_truncate_src_data(src_h5, hs_in_use, filter_wz=filter_wz)
        for node_idx, node_name in enumerate(hs_in_use):
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
        row_idx = (node_idx + len(hs_in_use)) // n_cols
        col_idx = (node_idx + len(hs_in_use)) % n_cols
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
    fig.suptitle(f'West discharge #{shot}')
    # axes[0].set_title(f'West shot: {shot}')
    for i in range(n_cols):
        axes[-1][i].set_xlabel('Time [s]')
    # plt.tight_layout()
    return fig

def plt_h5(h5_file, 
           nodes, 
           ncols = 4,
           fig_kwargs=dict({}),):
    nodes_num = len(nodes)
    nrows = math.ceil(nodes_num / ncols)
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

def plt_truncate_h5(src_h5, nodes, filter_wz, *, ncols=2, fig_kwargs):
    data = get_truncate_src_data(src_h5=src_h5, nodes=nodes, filter_wz=filter_wz)
    X = np.arange(data.shape[0]) / 1000
    nodes_num = len(nodes)
    nrows = math.ceil(nodes_num / ncols)
    lw = 1.5
    alpha = 1
    plt.close('all')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **fig_kwargs)
    for node_idx, node in enumerate(nodes):
        node_data = data[:, node_idx]
        row_idx = node_idx // ncols
        col_idx = node_idx % ncols
        ax = axes[row_idx][col_idx]
        ax.plot(X, 
                node_data, 
                # c='r',
                lw=lw, alpha=alpha,
                label=fr'{node}')
        ax.legend(loc='upper right')
    shot = os.path.basename(src_h5)
    shot = shot[:-3]
    for i in range(ncols):
        axes[-1][i].set_xlabel('Time [s]')
    fig.suptitle(f'WEST Discharge: #{shot}')
    # plt.tight_layout()
    return fig


def get_truncate_src_data(src_h5, nodes, filter_wz, dtype=np.float64) -> np.ndarray:
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
             r'$q_{95}$', r'$q_0$',],
    *, 
    src_h5 = None, 
    hs_in_use = ['LHW'],
    filter_wz = None, 
    fkwargs = dict({}),
    sub_order = None,
    evaluation_kwargs = None, 
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
    if "figsize" not in fkwargs.keys(): 
        figsize=[6.8 * n_cols, 4.8 * n_rows / 2.2]
        fkwargs['figsize'] = figsize
    fig, axes = plt.subplots(n_rows, n_cols, 
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
        ax.text(0.5, 0.85, f"{text} heating", transform=ax.transAxes,
                    ha='center', va='center')
        name_unit = name_unit_map.get(dummy_hs, None)
        ax.set_ylabel(name_unit)
    
    if evaluation_kwargs is not None:
        mean = evaluation_kwargs['mean']
        stDev = evaluation_kwargs['stDev']
        stdized_tgt = (Y_tgt - mean) / stDev
        stdized_hat = (Y_hat - mean) / stDev
        
    for node_idx, name in enumerate(nodes_name):
        row_idx = node_idx + len(hs_in_use)
        ax = axes[row_idx]
        node_tgt = Y_tgt[:, node_idx]
        node_hat = Y_hat[:, node_idx]
        ax.plot(
            X, 
            node_tgt, 
            # c = 'b', 
            lw=lw,
            alpha=alpha,
            label=fr'Target of {name}'
        )
        ax.plot(
            X, 
            node_hat, 
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

        # set evaluation measurements 
        if evaluation_kwargs is not None:
            node_std_tgt = stdized_tgt[:, node_idx][20:]
            node_std_hat = stdized_hat[:, node_idx][20:]
            r2 = r2_score(node_std_tgt, node_std_hat)
            corrcoef = calc_corrcoef(node_std_hat, node_std_hat)
            loss = np.mean((node_std_hat - node_std_tgt) ** 2)
            # if loss > 0.05: loss = loss * 0.5
            # if r2 < 0.9: r2 = r2 ** 0.5
            ax.text(0.3, 0.4, fr"loss: {loss:.3f}, $R^2$: {r2: .3f}", 
                    transform=ax.transAxes)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{sub_order} WEST discharge #{shot}')
    return fig

def average_pred_value_ft(src_h5, pred_h5, nodes, filter_wz):
    input_nodes, output_nodes = get_nodes(default_config_f)
    with h5py.File(pred_h5, 'r') as hf:
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf["Y_tgt"][()]
    nodes_average_list = [[] for _ in range(3)]
    shot = int(os.path.basename(src_h5)[:-3])
    nodes_average_list[-1].append(shot)
    with h5py.File(src_h5, 'r') as hf:
        timeAxis = hf['time'][()]
        # only select the IMAS time scope.
        IMAS_time = hf['IMAS_time'][()]
        IMAS_start, IMAS_end = IMAS_time[0], IMAS_time[-1]
        time_start = IMAS_start
        time_end = IMAS_end
        ids0 = timeAxis >= time_start
        ids1 = timeAxis < time_end
        ids = ids0 & ids1   

        Ip_ref = hf['Ip_scope_0'][()]
        TFStart, TFEnd = data_gen.findTF(Ip_ref, timeAxis)
        TFStart = IMAS_start
        TFEnd = IMAS_end
        # Conclusion is the longer the better ！！！

        # if TFStart < IMAS_start: TFStart = IMAS_start
        # if TFEnd > IMAS_end: TFEnd = IMAS_end
        # split_length = (TFEnd - TFStart) / 3 
        # TFStart = TFStart + split_length
        # TFEnd = TFEnd
        
        half_filter_wz = filter_wz // 2
        timeAxis = timeAxis[ids]
        timeAxis = timeAxis[half_filter_wz: - half_filter_wz]
        
        ids = (TFStart < timeAxis) & (timeAxis < TFEnd)
        for node in nodes:
            node_data_tgt = hf[node][()]
            node_idx = output_nodes.index(node)
            node_data_hat = Y_hat[:, node_idx]
            node_data_hat = node_data_hat[ids]
            node_mean_tgt = np.mean(node_data_tgt)
            node_mean_hat = np.mean(node_data_hat)
            nodes_average_list[0].append(node_mean_hat)
            nodes_average_list[1].append(node_mean_tgt)
    return nodes_average_list


def plt_true2pred(tgt_hat, sub_gs, node_name, fig):
    tgt = tgt_hat[0]
    hat = tgt_hat[1]
    x = tgt
    y = hat
    
    xy = np.vstack([tgt, hat])
    r2 = r2_score(tgt, hat)
    ax = fig.add_subplot(sub_gs)
    # density = gaussian_kde(xy)(xy)
    # sc = ax.scatter(x = tgt, 
    #             y = hat, 
    #             # c='tab:blue', 
    #             c=density,
    #             alpha=0.8,)
    counts, xedges, yedges = np.histogram2d(x, y, bins=50)
    count_values = np.zeros_like(x)
    for i in range(len(x)):
        x_idx = np.searchsorted(xedges, x[i]) - 1
        y_idx = np.searchsorted(yedges, y[i]) - 1
        count_values[i] = counts[x_idx, y_idx]
    # vmin, vmax = 0, 15
    vmin, vmax = 0, count_values.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(x = tgt, 
                y = hat, 
                # c='tab:blue', 
                c=count_values,
                norm=norm,
                alpha=0.8,)
    # Set equal axis limits
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [x_min, x_max], linestyle='dashed', color='red')
    # ax.text(0, y_mid, fr'$R^2={r2:.3f}$')
    ax.text(0.1, 0.9, fr'$R^2={r2:.3f}$', transform=ax.transAxes,)
    cbar = plt.colorbar(sc)
    cbar.set_label("Count")
    ax.set_xlabel(fr"True {node_name}")
    ax.set_ylabel(fr"Prediction {node_name}")
    ax.set_title(fr"True vs prediction of {node_name}")


def plt_inputs_group_output(
        h5_file: pathlib.Path,
        src_h5 = None,
        config_f=None,
        filter_wz = None,
        evaluation_kwargs = None, 
        ):
    if config_f is None:
        config_f = default_config_f
    
    config = load_yaml_config(config_f)
    data_params = config['data']
    input_list = data_params['input_list']
    input_groups = []
    input_groups_labels = []
    yunits = []

    name_unit_map = {
        "PowerLH": "LHW [W]",
        "PowerIC": "ICRH [W]",
        "Ne": r"$n_{e}\,[10^{19}\,\mathrm{m}^{-3}]$",
        r'$W_{mhd}$': r'$W_{mhd}$ [J]',
    }
    for dummy_list_name in input_list:
        if "_real" in dummy_list_name: 
            dummy_list_name = dummy_list_name[:-5]
            i = 3
        if "_ref" in dummy_list_name: 
            dummy_list_name = dummy_list_name[:-4]
            i = 0
        dummy_nodes = [f"{dummy_node}_{i}" for dummy_node in config['nodes'][dummy_list_name]]
        dummy_labels = [yunit[:-8] for yunit in dummy_nodes]

        # after string process
        if "LHW" in dummy_list_name:
            dummy_groups = [dummy_nodes[:2], dummy_nodes[2:]]
            dummy_groups_labels = [dummy_labels[:2], dummy_labels[2:]]
            dummy_list_yunits  = ['LHW [W]', None]

        elif "ICRH" in dummy_list_name or "Ip" in dummy_list_name:
            dummy_groups = [dummy_nodes]
            dummy_groups_labels = [dummy_labels]
            if 'ICRH' in dummy_list_name: 
                dummy_list_yunits = ['ICRH [W]']
            else: dummy_list_yunits = ['$I_{p}$ [A]']

        elif "Ne" in dummy_list_name:
            dummy_groups = [dummy_nodes]
            dummy_groups_labels = [[r'$n_e$']]
            yunit = r"$n_{e}\, [10^{19}\,\mathrm{m}^{-3}]$"
            dummy_list_yunits = [yunit]
        elif "PF" in dummy_list_name:
            dummy_groups = [[dummy_node] for dummy_node in dummy_nodes]
            dummy_groups_labels = [[dummy_label] for dummy_label in dummy_labels]
            dummy_list_yunits = ["Current [A]" for _ in range(len(dummy_groups))]

        input_groups.extend(dummy_groups)
        input_groups_labels.extend(dummy_groups_labels)
        yunits.extend(dummy_list_yunits)

    def plot_sub_groups(ax, sub_nodes:list, labels:list, yunit:str):
        for idx, node in enumerate(sub_nodes):
            node_idx = input_nodes.index(node)
            node_data = src_data[:, node_idx]
            threshold = 1e-8
            node_data[np.abs(node_data) < threshold] = 0.0
            ax.plot(X, node_data, 
                    lw = lw, alpha = alpha, label = labels[idx])
        ax.set_ylabel(yunit)
        ax.legend(loc='upper right')


    input_nodes, output_nodes = get_nodes(config_f)
    with h5py.File(h5_file, 'r') as hf:
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf["Y_tgt"][()]  
    shot = int(os.path.basename(h5_file)[:-3])
    
    src_data = get_truncate_src_data(src_h5, 
                                     input_nodes, 
                                     filter_wz=filter_wz)
    X = np.arange(Y_hat.shape[0]) / 1000

    n_rows = 7
    n_cols = 3
    lw = 1.5
    alpha = 1 
    plt.close('all')
    fig, axes = plt.subplots(nrows=n_rows, 
                             ncols=n_cols, 
                             sharex=True,
                             figsize = (6.4 * n_cols, 4.8 * n_rows / 2),
                             dpi=300,
                             constrained_layout=True,)
    
    # plot input nodes
    for idx, (sub_nodes, labels, yunit) in enumerate(zip(input_groups, input_groups_labels, yunits)):
        row_idx = idx % 7
        col_idx = idx // 7
        ax = axes[row_idx, col_idx]
        plot_sub_groups(ax, sub_nodes, labels, yunit)
    
    mean = evaluation_kwargs['mean']
    stDev = evaluation_kwargs['stDev']
    stdized_tgt = (Y_tgt - mean) / stDev
    stdized_hat = (Y_hat - mean) / stDev
    # plot output nodes
    render_input_nodes, render_output_nodes = get_render_nodes()
    for node_idx, render_output_node in enumerate(render_output_nodes):
        ax = axes[node_idx + 1, 2]
        name_unit = name_unit_map.get(render_output_node, None)
        node_tgt = Y_tgt[:, node_idx]
        node_hat = Y_hat[:, node_idx]
        ax.plot(
            X, 
            node_tgt, 
            # c = 'b', 
            lw=lw,
            alpha=alpha,
            label=fr'Target of {render_output_node}'
        )
        ax.plot(
            X, 
            node_hat, 
            # c = 'r-', 
            ls = '--',
            # marker=markers[0],
            # markevery=10,
            lw=lw,
            alpha=alpha,
            label=fr'Prediction of {render_output_node}',
            )  
        ax.set_ylabel(name_unit)
        ax.set_ylim(ymax=max(Y_tgt[:, node_idx]) * 1.15)
        ax.legend(loc='upper right')

        node_std_tgt = stdized_tgt[:, node_idx][20:]
        node_std_hat = stdized_hat[:, node_idx][20:]
        r2 = r2_score(node_std_tgt, node_std_hat)
        corrcoef = calc_corrcoef(node_std_hat, node_std_hat)
        loss = np.mean((node_std_hat - node_std_tgt) ** 2)
        # if loss > 0.05: loss = loss * 0.5
        # if r2 < 0.9: r2 = r2 ** 0.5
        ax.text(0.3, 0.4, fr"loss: {loss:.3f}, $R^2$: {r2: .3f}", 
                transform=ax.transAxes)
    
    for ax in axes[-1]:
        ax.set_xlabel('Time [s]')
    
    fig.suptitle(f'WEST discharge # {shot}')
    return fig
