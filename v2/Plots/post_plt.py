#  -*- coding: utf-8 -*-
import os
import pathlib
import h5py

# from private_modules.utilities import 

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import seaborn as sns
import math

from sklearn.metrics import r2_score

from private_modules.utilities import load_yaml_config, convert_hdf5_2dict
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from scipy.io import loadmat


def calc_metrics(h5_file, metric_names, output_mean_stds):
    metrics = dict({})
    with h5py.File(h5_file) as hf:
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf['Y_tgt'][()]
    
    Y_hat = (Y_hat - output_mean_stds[0, :]) / output_mean_stds[1, :]
    Y_tgt = (Y_tgt - output_mean_stds[0, :]) / output_mean_stds[1, :]
    # calc metrics
    if 'r2' in metric_names:
        r2 = r2_score(Y_tgt, Y_hat)
        metrics['r2'] = r2
    if 'mse' in metric_names:
        mse = np.sqrt(np.mean((Y_hat - Y_tgt) ** 2))
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
    names = kwargs['name']
    for ix in range(len(r2s_list)):
        name = names[ix]
        r2s = r2s_list[ix]
        row = ix // n_cols
        col = ix % n_cols
        sub_gs = gs[row, col]
        plt_func(r2s, sub_gs, name, fig)
        # sub = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[row, col], wspace=0)
        # ax0 = fig.add_subplot(sub[:, :-1])
        # ax1 = fig.add_subplot(sub[:, -1])
    plt.tight_layout()
    return fig

def scatter_r2s(r2s, sub_gs, name, fig):
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
    ax0.set_ylabel(r"$R^2$")
    ax0.set_ylim(0, 1)

    ax1.set_ylim(0, 1)
    sns.histplot(y=r2s, color="b", kde=True, ax=ax1, linewidth=0)
    ax1.set_xlabel("")
    avg_sim = np.mean(r2s)
    ax0.set_title(fr"{name} $R^2$ is {avg_sim:.3f}")

def plt_h5(h5_file, nodes):
    nodes_num = len(nodes)
    ncols = 2 
    nrows = math.ceil(nodes_num / ncols)
    fig_kwargs = {"constrained_layout": False,
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
            row_idx = node_idx // 2
            col_idx = node_idx % 2
            ax = axes[row_idx][col_idx]
            ax.plot(time, node_data, c='r', lw=lw, alpha=alpha,
                    label=fr'{node}')
            ax.legend(loc='upper right')
    shot = os.path.basename(h5_file)
    shot = shot[:-3]
    axes[0][-1].set_xlabel('Time [s]')
    axes[1][-1].set_xlabel('Time [s]')
    fig.suptitle(f'Shot: {shot}')
    plt.tight_layout()
    return fig

def plot_all_scopes(mat_file_path):
    data = loadmat(mat_file_path)
    mat_file_path = pathlib.Path(mat_file_path)
    shot = mat_file_path.stem.split('_')[-1]
    scope_keys = [key for key in data.keys() if key.endswith('_scope')]
    # with open("a.txt", 'w') as f:
    #     for scope_key in scope_keys:
    #         f.write(f'"{scope_key}", ')
    print(scope_keys)
    n_input_dim = len(scope_keys) * 4
    num_signals = 4
    ncols = num_signals
    fig_kwargs = {"constrained_layout": False,
                 "figsize": (6.8 * ncols, 4.8 * math.ceil(n_input_dim / 4) ),
                 "dpi": 300}
    plt.close('all')
    fig, axes = plt.subplots(nrows=math.ceil(n_input_dim / ncols), ncols=ncols, **fig_kwargs)
    for row_idx, scope_name in enumerate(scope_keys):
        # Each scope is expected to be a struct at data[scope_name][0,0]
        scope_struct = data[scope_name][0,0]
        time_field = scope_struct['time'].squeeze()
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0,0]  
        
        num_signals = signal_values.shape[1]
        for col_idx in range(num_signals):
            ax = axes[row_idx, col_idx]
            ax.plot(time_field, signal_values[:, col_idx])
            ax.set_title(f'{scope_name} Signal {col_idx}')
            if "Vloop" in scope_name:
                ax.set_ylim(-2, 1)
            ax.set_xlabel('Time')
            ax.set_ylabel('Signal Amplitude')
    plt.tight_layout()
    return fig

def calc_mean_p_matrix(p_matrix_f):
    """  calc mean of p_matrix_list
    """
    p_matrix_list = np.load(p_matrix_f)
    nrows = p_matrix_list.shape[1]
    ncols = p_matrix_list.shape[2]
    p_matrix = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            p_matrix[i, j] = np.mean(np.abs(p_matrix_list[:, i, j][~np.isnan(p_matrix_list[:, i, j])]))
    return p_matrix


def plt_pred_shot(
        h5_file, 
        nodes_name=[r'$\beta_{n}$', r'$\beta_{p}$', 
                    r'$\beta_{t}$', r'$w_{mhd}$', 
                    r'$q_0$', r'$q_{95}$']):
    with h5py.File(h5_file, 'r') as hf:
        keys = list(hf.keys())
        Y_hat = hf['Y_hat'][()]
        Y_tgt = hf["Y_tgt"][()]
    shot = int(os.path.basename(h5_file)[:-3])
    plt.close('all')
    lw = 1.5
    alpha = 1
    markers = ['D', '*', 'X']
    fig, axes = plt.subplots(3, 2, 
                             figsize=[6.8 * 2, 4.8 * 2], 
                             # dpi=300, 
                             sharex=True)
    X = np.arange(Y_hat.shape[0]) / 1000
    for node_idx, name in enumerate(nodes_name):
        row_idx = node_idx // 2
        col_idx = node_idx % 2
        ax = axes[row_idx][col_idx]
        ax.plot(
            X, 
            Y_tgt[:, node_idx], 
            c = 'r', 
            lw=lw,
            alpha=alpha,
            label=fr'Target of {name}'
        )
        ax.plot(
            X, 
            Y_hat[:, node_idx], 
            c = 'b', 
            # marker=markers[0],
            # markevery=10,
            lw=lw,
            alpha=alpha,
            label=fr'Prediction of {name}')
        # if node_idx == 3:
        ax.set_ylim(ymax=max(Y_tgt[:, node_idx]) * 1)
        ax.legend(loc='upper right')
    fig.suptitle(f'West shot: {shot}')
    # axes[0].set_title(f'West shot: {shot}')
    axes[0][-1].set_xlabel('Time [s]')
    axes[1][-1].set_xlabel('Time [s]')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    pass