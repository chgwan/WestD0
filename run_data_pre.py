# -*- coding: utf-8 -*-
"""
Project-specific data preprocessing workflows for WEST D0.

Thin script that wires project config / paths to reusable library
functions in ``src.data_utils``.

Usage:
    python run_data_pre.py <func> [options]

    python run_data_pre.py save_merge
    python run_data_pre.py clear_h5
    python run_data_pre.py stat_nodes_h5
    python run_data_pre.py mp_h5_p_matrix --option gp --maxlag 4
    python run_data_pre.py plt_p_matrix --option gp
"""
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from src.data_utils import (
    merge_mat_h5,
    filter_h5,
    strongest_granger_causality,
    strongest_z_score,
)
from src.utils import load_yaml_config, save_to_file, screen_print
from src.proj_config import get_proj_config


# ---------------------------------------------------------------------------
# Directories (all derived from proj_config / base.yml)
# ---------------------------------------------------------------------------

_cfg = get_proj_config()
PCS_DIR = _cfg.pcs_dir
IMASH5_DIR = _cfg.imash5_dir
WESTDATA_DIR = _cfg.data_dir
ERROR_DIR = _cfg.error_dir
STAT_DIR = _cfg.stat_dir
FIGS_DIR = _cfg.figs_dir


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_nodes(config_f=None):
    """Parse input/output node lists from the YAML config."""
    proj_cfg = get_proj_config()
    if config_f is None:
        config = proj_cfg.base_config
    else:
        base = proj_cfg.base_config
        override = load_yaml_config(config_f)
        config = {**base, **override}
    data_params = config["data"]
    input_list = data_params["input_list"]
    output_list = data_params["output_list"]
    input_nodes, output_nodes = [], []
    for name in input_list:
        if "_real" in name:
            name = name[:-5]
            suffix = 3
        elif "_ref" in name:
            name = name[:-4]
            suffix = 0
        else:
            suffix = None
        if suffix is not None:
            input_nodes.extend(
                [f"{n}_{suffix}" for n in config["nodes"][name]])
        else:
            input_nodes.extend(config["nodes"][name])
    for name in output_list:
        output_nodes.extend(config["nodes"][name])
    return input_nodes, output_nodes


_RENDER_INPUT_MAP = {
    "REF1_scope": "Ne",
    "Ip_scope": "Ip_ref",
}


def get_render_nodes():
    """Return human-readable labels for plotting."""
    config = get_proj_config().base_config
    data_params = config["data"]
    input_list = data_params["input_list"]
    raw_input_nodes = []
    for name in input_list:
        if "_real" in name:
            name = name[:-5]
        elif "_ref" in name:
            name = name[:-4]
        raw_input_nodes.extend(config["nodes"][name])
    input_labels = []
    for node in raw_input_nodes:
        if node in _RENDER_INPUT_MAP:
            input_labels.append(_RENDER_INPUT_MAP[node])
        else:
            input_labels.append(node.removesuffix("_scope"))
    output_labels = config["nodes"]["rendered_D0_signals"]
    return input_labels, output_labels


# ---------------------------------------------------------------------------
# Concurrent helpers
# ---------------------------------------------------------------------------

def mp_run(func, items, num_workers=None, desc=None):
    """Run func on each item with a ProcessPoolExecutor (no return values)."""
    if num_workers is None or num_workers == -1:
        num_workers = os.cpu_count()
    if desc is None:
        desc = getattr(func, '__name__', 'processing')
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()


def mp_map(func, items, num_workers=None, desc=None):
    """Run func on each item with a ProcessPoolExecutor, return results list."""
    if num_workers is None or num_workers == -1:
        num_workers = os.cpu_count()
    if desc is None:
        desc = getattr(func, '__name__', 'processing')
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        # Collect in submission order
        ordered = [None] * len(items)
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx = futures[future]
            ordered[idx] = future.result()
        results = ordered
    return results


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

class StatH5Dir:
    def __init__(self, num_workers=None, eps=1e-7):
        if num_workers is None or num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.eps = eps

    def stat_dir(self, h5s_dir, nodes, stat_h5_path):
        h5s_dir = pathlib.Path(h5s_dir)
        self.h5s_dir = h5s_dir
        stat_h5_path = pathlib.Path(stat_h5_path)
        shots = [int(f.stem) for f in h5s_dir.glob("*.h5")]
        h5_files = [h5s_dir / f"{shot}.h5" for shot in shots]
        df = self._stat_files(h5_files, nodes)
        df.to_csv(stat_h5_path)
        return df

    def _stat_files(self, h5_files, nodes):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._stat_file, f, nodes): f
                for f in h5_files
            }
            results = []
            with tqdm(as_completed(futures), total=len(futures),
                      desc="Computing stats") as pbar:
                for future in pbar:
                    pbar.set_postfix_str(pathlib.Path(futures[future]).name)
                    results.append(future.result())
        df = pd.DataFrame(results).set_index("shot")
        return df

    def _stat_file(self, h5_file, nodes):
        stat_dict = {}
        with h5py.File(h5_file, mode="r") as hf:
            try:
                shot = hf["shot"][()]
            except KeyError:
                shot = int(pathlib.Path(h5_file).stem)
            stat_dict["shot"] = shot
            keys = list(hf.keys())

            for node in nodes:
                if node not in keys:
                    self._set_invalid(stat_dict, node)
                    continue
                dataset = hf[node]
                if dataset.shape is None or len(dataset.shape) == 0:
                    self._set_invalid(stat_dict, node)
                    continue
                data = dataset[()]
                time_key = f"{node}_time" if f"{node}_time" in keys else "time"
                times = hf[time_key][()]
                if times.shape is None:
                    self._set_invalid(stat_dict, node)
                    continue
                valid = np.isfinite(data) if data.ndim == 1 else np.ones(len(data), dtype=bool)
                valid_data = data[valid]
                stat_dict[node] = True
                stat_dict[f"{node}_sum"] = np.sum(valid_data, dtype=np.float64)
                stat_dict[f"{node}_square_sum"] = np.sum(valid_data ** 2, dtype=np.float64)
                stat_dict[f"{node}_length"] = len(valid_data)
        return stat_dict

    @staticmethod
    def _set_invalid(stat_dict, node):
        stat_dict[node] = False
        stat_dict[f"{node}_sum"] = None
        stat_dict[f"{node}_square_sum"] = None
        stat_dict[f"{node}_length"] = 0

    def calc_MS(self, nodes, stat_file, MS_file=None):
        stat_df = pd.read_csv(stat_file, index_col=0, low_memory=False)
        means = np.full(len(nodes), np.nan, dtype=np.float64)
        stdevs = np.full(len(nodes), np.nan, dtype=np.float64)

        for i, node in enumerate(nodes):
            square_sums = stat_df[f"{node}_square_sum"].values.astype(np.float64)
            sums = stat_df[f"{node}_sum"].values.astype(np.float64)
            existence = stat_df[node].values.astype(bool)
            lengths = stat_df[f"{node}_length"].values

            valid = np.isfinite(square_sums) & existence
            if not np.any(valid):
                continue

            N = np.sum(lengths[valid], dtype=np.int64)
            if N == 0:
                continue
            S = np.sum(sums[valid])
            S2 = np.sum(square_sums[valid])
            mean = S / N
            variance = (S2 - S ** 2 / N) / (N - 1)
            stdev = np.sqrt(max(0, variance))
            means[i] = float(mean)
            stdevs[i] = float(stdev)

        MS_df = pd.DataFrame({"mean": means, "stDev": stdevs}, index=nodes).T
        if MS_file is not None:
            MS_df.to_csv(MS_file)
        return MS_df


# ---------------------------------------------------------------------------
# Step 1: Merge IMAS H5 + DCS .mat -> WestData
# ---------------------------------------------------------------------------

def _merge_one(shot):
    h5_file = IMASH5_DIR / f"{shot}.h5"
    mat_file = PCS_DIR / f"DCS_archive_{shot}.mat"
    if not mat_file.exists():
        print(f"[SKIP] No .mat for shot {shot}")
        return
    data_dict = merge_mat_h5(h5_file, mat_file)
    save_to_file(WESTDATA_DIR / f"{shot}.h5", data_dict, is_overwrite=True)


def save_merge():
    """Merge IMAS H5 files with DCS .mat files -> WestData/."""
    WESTDATA_DIR.mkdir(exist_ok=True)
    shots = [int(f.stem) for f in IMASH5_DIR.glob("*.h5")]
    screen_print(f"Merging {len(shots)} shots")
    mp_run(_merge_one, shots, num_workers=10, desc="Merging")
    screen_print("save_merge done")


# ---------------------------------------------------------------------------
# Step 2: Move invalid shots to ErrorShots/
# ---------------------------------------------------------------------------

def _check_h5(h5):
    if filter_h5(h5):
        shutil.move(str(h5), str(ERROR_DIR))


def clear_h5():
    """Move invalid shots to ErrorShots/."""
    ERROR_DIR.mkdir(exist_ok=True)
    h5s = list(WESTDATA_DIR.glob("*.h5"))
    screen_print(f"Checking {len(h5s)} shots")
    mp_run(_check_h5, h5s, num_workers=10, desc="Filtering")
    screen_print("clear_h5 done")


# ---------------------------------------------------------------------------
# Step 3: Compute per-node statistics and global mean/stDev
# ---------------------------------------------------------------------------

def stat_nodes_h5():
    """Compute per-node statistics and global mean/stDev."""
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    h5_stat_csv = STAT_DIR / "h5_stat.csv"

    tmp_h5 = next(WESTDATA_DIR.glob("*.h5"))
    with h5py.File(tmp_h5, "r") as hf:
        keys = [k for k in hf.keys() if "time" not in k]

    stat_runner = StatH5Dir(num_workers=40)
    stat_runner.stat_dir(WESTDATA_DIR, keys, h5_stat_csv)

    MS_file = STAT_DIR / "h5_global_MS.csv"
    stat_runner.calc_MS(keys, h5_stat_csv, MS_file)
    screen_print("stat_nodes_h5 done")


# ---------------------------------------------------------------------------
# Step 4: Granger / correlation analysis
# ---------------------------------------------------------------------------

def _read_h5_raw(h5_file, nodes, dtype=np.float32):
    """Simple H5 reader for Granger/correlation analysis (no filtering)."""
    with h5py.File(h5_file, "r") as hf:
        time_axis = hf["time"][()]
        n_time = len(time_axis)
        data = np.empty((n_time, len(nodes)), dtype=dtype)
        node_flags = np.ones(len(nodes), dtype=np.int32)
        for j, node in enumerate(nodes):
            if node in hf:
                arr = hf[node][()]
                if isinstance(arr, h5py.Empty) or arr is None:
                    data[:, j] = 0.0
                    node_flags[j] = 0
                else:
                    data[:, j] = arr.astype(dtype)
            else:
                data[:, j] = 0.0
                node_flags[j] = 0
    return data, node_flags, time_axis


def _calc_gp(input_arr, output_arr, node_flags, option="gp", maxlag=4):
    input_dim = input_arr.shape[1]
    output_dim = output_arr.shape[1]
    p_matrix = np.zeros((input_dim, output_dim))
    for in_idx, sig_in in enumerate(input_arr.T):
        unique = np.unique(sig_in.round(decimals=4))
        if node_flags[in_idx] == 0 or len(unique) == 1:
            p_matrix[in_idx, :] = np.nan
            continue
        for out_idx, sig_out in enumerate(output_arr.T):
            unique = np.unique(sig_out.round(decimals=4))
            if node_flags[input_dim + out_idx] == 0 or len(unique) < 10:
                p_matrix[:, out_idx] = np.nan
                continue
            if option == "gp":
                p_matrix[in_idx, out_idx] = strongest_granger_causality(
                    sig_out, sig_in, maxlag=maxlag)
            elif option == "zp":
                p_matrix[in_idx, out_idx] = strongest_z_score(
                    sig_out, sig_in, maxlag)
    return p_matrix


def _h5_p_matrix(h5_file, option="gp", maxlag=4):
    input_nodes, output_nodes = get_nodes()
    nodes = list(input_nodes) + list(output_nodes)
    data, node_flags, _ = _read_h5_raw(h5_file, nodes)
    input_dim = len(input_nodes)
    return _calc_gp(
        data[:, :input_dim], data[:, input_dim:],
        node_flags, option=option, maxlag=maxlag)


def mp_h5_p_matrix(num_workers=None, option="gp", maxlag=4):
    """Compute Granger/correlation matrices across all shots."""
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    h5s = list(WESTDATA_DIR.glob("*.h5"))
    worker_fn = partial(_h5_p_matrix, option=option, maxlag=maxlag)
    p_matrix_list = mp_map(worker_fn, h5s, num_workers=num_workers,
                           desc=f"Granger ({option})")
    p_arr = np.stack(p_matrix_list)
    np.save(STAT_DIR / f"p_matrix_{option}.npy", p_arr)
    screen_print(f"mp_h5_p_matrix ({option}) done")


# ---------------------------------------------------------------------------
# Step 5: Plot the matrices
# ---------------------------------------------------------------------------

def plt_p_matrix(option="gp"):
    """Plot Granger causality or correlation heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8-poster")
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    p_file = STAT_DIR / f"p_matrix_{option}.npy"
    p_list = np.load(p_file)
    data = np.nanmean(np.abs(p_list), axis=0)

    title = "(a) Granger causality" if option == "gp" else "(b) Absolute correlation coefficient"
    fig_name = "granger_causality" if option == "gp" else "correlation_coefficient"

    input_labels, output_labels = get_render_nodes()
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    fig = plt.figure(figsize=(10, 10), dpi=200)
    sns.heatmap(
        data, annot=True, fmt=".2e", cmap=cmap,
        xticklabels=output_labels, yticklabels=input_labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Score"},
        annot_kws={"size": "large"})
    plt.title(f"{title} between input and output signals", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"{fig_name}.pdf")
    screen_print(f"Saved {fig_name}.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_FUNCS = {
    "save_merge":      save_merge,
    "clear_h5":        clear_h5,
    "stat_nodes_h5":   stat_nodes_h5,
    "mp_h5_p_matrix":  mp_h5_p_matrix,
    "plt_p_matrix":    plt_p_matrix,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="WEST D0 data preprocessing workflows")
    parser.add_argument("func", choices=_FUNCS, help="Function to run")

    # mp_h5_p_matrix / plt_p_matrix
    parser.add_argument("--option", type=str, default="gp",
                        choices=["gp", "zp"], help="Analysis type")
    parser.add_argument("--maxlag", type=int, default=4,
                        help="Max lag for Granger causality")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPUs)")

    args = parser.parse_args()

    func = _FUNCS[args.func]
    if args.func == "mp_h5_p_matrix":
        func(num_workers=args.num_workers, option=args.option, maxlag=args.maxlag)
    elif args.func == "plt_p_matrix":
        func(option=args.option)
    else:
        func()
