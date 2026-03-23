# -*- coding: utf-8 -*-
"""
WEST D0 distributed training entry point.

Launch with torchrun:
    torchrun --nproc_per_node=4 run_train.py --config configs/former.yml

Single-GPU:
    python run_train.py --config configs/former.yml

Restore from checkpoint:
    torchrun --nproc_per_node=4 run_train.py --config configs/former.yml --train-type restore
"""
import argparse
import os
import pathlib
import time
import random

import torch
from torch import optim
import pandas as pd
from functools import partial

from src import data_gen, mlmodels, model_dist
from src.utils import load_yaml_config, deep_merge, clean_dir, screen_print, random_init
from src.utils import calc_loss_Former, calc_loss_RNN
from src.proj_config import get_proj_config
from run_data_pre import get_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPTIMIZER_MAP = {
    "torch.optim.SGD": torch.optim.SGD,
    "torch.optim.Adam": torch.optim.Adam,
    "torch.optim.AdamW": torch.optim.AdamW,
}

MODEL_MAP = {
    "MLP": mlmodels.MLP,
    "FastLSTM": mlmodels.FastLSTM,
    "Former": mlmodels.WestFormer,
    "ERT": mlmodels.WestERT,
    "GPT": mlmodels.WestGPT,
}

LOSS_MAP = {
    "MLP": calc_loss_Former,
    "FastLSTM": calc_loss_RNN,
    "Former": calc_loss_Former,
    "ERT": calc_loss_Former,
    "GPT": calc_loss_Former,
}


def _my_OneCycleLR(optimizer, max_lr, total_steps):
    return optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    proj_cfg = get_proj_config()
    base_config = proj_cfg.base_config
    model_config = load_yaml_config(args.config)
    config = deep_merge(base_config, model_config)

    train_params = config["train"]
    model_params = config["model"]
    data_params = config["data"]
    base_dir = os.path.expandvars(config['base_dir'])
    random_init(seed=train_params['random_seed'])

    # torchrun sets WORLD_SIZE; fallback to GPU count for single-node
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    rank = int(os.environ.get('RANK', 0))

    data_dir = proj_cfg.data_dir
    stat_dir = proj_cfg.stat_dir

    stat_f = stat_dir / "h5_stat.csv"
    stat_df = pd.read_csv(stat_f, index_col=0)

    h5s = list(data_dir.glob('*.h5'))

    MS_file = stat_dir / 'h5_global_MS.csv'
    MS_df = pd.read_csv(MS_file, index_col=0)

    input_nodes, output_nodes = get_nodes()

    sample_num = data_params['sample_num']
    if isinstance(sample_num, int):
        h5s = h5s[:sample_num]
    if rank == 0:
        screen_print(f'{len(h5s)} h5 files input')

    if data_params['shuffle']:
        random.shuffle(h5s)

    filter_wz = data_params.get('filter_wz', 99)
    is_output_filter = train_params.get('is_output_filter', True)
    non_filter_nodes = [] if is_output_filter else output_nodes

    my_data_gen = data_gen.H5GenDataLoader(
        h5s=h5s,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        batch_size=data_params['batch_size'] * world_size,
        num_workers=data_params['num_workers'],
        DS=data_gen.StdWESTShotDS,
        MS_df=MS_df,
        stat_df=stat_df,
        pin_memory=True,
        world_size=world_size,
        filter_func=data_gen.SMA,
        filter_wz=filter_wz,
        non_filter_nodes=non_filter_nodes,
    )
    ratio_list = data_params.get('ratios', [0.8, 0.1, 0.1])
    my_data_gen.set_split_ratio(ratio_list)

    num_epochs = train_params['num_epochs']
    train_params['optimizer_fn'] = OPTIMIZER_MAP[config['optimizer']['name']]
    train_params['learning_rate'] = float(config['optimizer']['lr']) * world_size

    train_type = args.train_type or train_params['train_type']

    train_base_dir = os.path.join(base_dir, config['summary']['root_dir'])
    if rank == 0:
        pathlib.Path(train_base_dir).mkdir(exist_ok=True)
        if train_type == "train":
            clean_dir(train_base_dir)

    if train_type == "restore":
        train_params['checkpoint_path'] = os.path.join(
            base_dir, train_params['checkpoint_path'])

    # Scheduler
    data_loaders = my_data_gen.sp_ratio_wz()
    tra_loaders = data_loaders[0]
    scheduler_fn = partial(
        _my_OneCycleLR,
        max_lr=min(train_params['learning_rate'] * 1e3, 1e-1),
        total_steps=(len(tra_loaders[0]) + tra_loaders[0].num_workers) * num_epochs,
    )
    train_params['scheduler_fn'] = scheduler_fn
    train_params.update(config['summary'])

    # Build model
    model_name = model_params['name']
    model_fn = MODEL_MAP[model_name]
    loss_fn = LOSS_MAP[model_name]
    model = model_fn(**model_params)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        screen_print(f"Total parameters: {total_params / 1e3:.0f} K")

    start = time.time()

    my_model_train = model_dist.ModelTrainTruncatedRNN(
        my_data_gen, num_epochs,
        loss_fn, train_base_dir, train_params,
        **model_params,
    )
    my_model_train.run_train(model, restore=(train_type == "restore"))

    end = time.time()
    if rank == 0:
        screen_print(f"Running time {end - start:.1f} seconds")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="WEST D0 distributed training")
    parser.add_argument("--config", type=str, required=True,
                        help="Model YAML config path (e.g. configs/former.yml)")
    parser.add_argument("--train-type", type=str, default='train',
                        choices=["train", "restore"],
                        help="Training mode; overrides config")
    return parser.parse_args()


if __name__ == "__main__":
    # Set CUDA_VISIBLE_DEVICES only when NOT managed by a scheduler
    if os.getenv('SLURM_JOB_ID') is None and os.getenv('PBS_JOBID') is None:
        proj_cfg = get_proj_config()
        cuda_devices = proj_cfg.base_config.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    main()
