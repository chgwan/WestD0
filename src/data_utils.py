# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import h5py
from statsmodels.tsa.stattools import grangercausalitytests


# ── Inlined from private_modules ─────────────────────────────────────

def convert_hdf5_2dict(file_name: os.PathLike) -> dict:
    data_dict = {}
    with h5py.File(file_name, 'r') as hf:
        for key in hf.keys():
            val = hf[key][()]
            if isinstance(val, h5py.Empty):
                val = None
            data_dict[key] = val
    return data_dict


def calc_corrcoef(arr_0, arr_1):
    """Correlation coefficient between two 1-D arrays."""
    mean_arr_0 = np.mean(arr_0)
    mean_arr_1 = np.mean(arr_1)
    nominator = np.sum((arr_0 - mean_arr_0) * (arr_1 - mean_arr_1))
    denominator = np.sqrt(np.sum((arr_0 - mean_arr_0) ** 2) * np.sum((arr_1 - mean_arr_1) ** 2))
    return nominator / denominator


# ── Data reading ─────────────────────────────────────────────────────

def read_all_scope(mat_file: os.PathLike):
    data = loadmat(mat_file)
    scope_keys = [key for key in data.keys() if key.endswith('_scope')]
    scope_dict = {}
    scope_struct = data['Ip_scope'][0, 0]
    time_field = scope_struct['time'].squeeze()
    Ip_scope_0 = scope_struct['signals']['values'][0, 0][:, 0]

    scope_dict['time'] = time_field
    ids = Ip_scope_0 > 2
    start_idx = min(np.arange(len(ids))[ids])
    start_time = time_field[start_idx]
    for scope_name in scope_keys:
        scope_struct = data[scope_name][0, 0]
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0, 0]
        for i in [0, 3]:
            signal_value = signal_values[:, i]
            scope_dict[f'{scope_name}_{i}'] = signal_value
    return scope_dict, start_time


def merge_mat_h5(h5, mat_file):
    """Merge IMAS h5 file with DCS mat file, aligned on discharge start."""
    mat_dict, start_time = read_all_scope(mat_file)
    dt = 32 - start_time
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
    """Find the end of the flat-top phase from Ip_ref."""
    IpMax = max(Ip_ref)
    third_Ip = IpMax // 3
    ids = Ip_ref > third_Ip
    d_Ip_ids = np.abs(np.gradient(Ip_ref, time)) < 500
    TFEnd = time[d_Ip_ids & ids][-1]
    return TFEnd


def filter_h5(h5):
    """Check the h5 validity."""
    with h5py.File(h5, 'r') as hf:
        Ip_ref = hf['Ip_scope_0'][()]
        time = hf['time'][()]
        TFEnd = findTFEnd(Ip_ref, time)

        IMAS_time = hf['IMAS_time'][()]
        IMAS_start, IMAS_end = IMAS_time[0], IMAS_time[-1]
        if IMAS_end - TFEnd < 0:
            return True
        else:
            return False


# ── Causality / correlation ──────────────────────────────────────────

def strongest_granger_causality(arr0, arr1, maxlag: int = 4):
    """Minimum p-value across all lags and test types for Granger causality from arr1 to arr0."""
    data = np.column_stack([arr0, arr1])
    result = grangercausalitytests(data, maxlag=maxlag)

    test_names = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
    min_pval = float('inf')

    for lag in result:
        for name in test_names:
            pval = result[lag][0][name][1]
            if pval < min_pval:
                min_pval = pval
    return min_pval


def strongest_z_score(arr0, arr1, maxlag: int = 4):
    """Maximum absolute correlation coefficient across lags."""
    zps = []
    for lag in range(maxlag + 1):
        if lag == 0:
            target_arr = arr0
            pred_arr = arr1
        else:
            target_arr = arr0[lag:]
            pred_arr = arr1[:-lag]
        zp = calc_corrcoef(target_arr, pred_arr)
        zps.append(np.abs(zp))
    return np.max(zps)


if __name__ == "__main__":
    read_all_scope("")
