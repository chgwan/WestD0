# -*- coding: utf-8 -*-
"""
WEST D0 distributed inference entry point.

Launch with torchrun:
    torchrun --nproc_per_node=4 run_infer.py --config configs/former.yml

Single-GPU:
    python run_infer.py --config configs/former.yml

Custom checkpoint:
    python run_infer.py --config configs/former.yml --checkpoint path/to/model.pt
"""
import argparse
import os
import pathlib
import random

import natsort
import torch
import pandas as pd

from src import data_gen, mlmodels, model_dist
from src.utils import load_yaml_config, deep_merge, screen_print, random_init
from src.utils import inference_fn
from src.proj_config import get_proj_config
from run_data_pre import get_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "MLP": mlmodels.MLP,
    "FastLSTM": mlmodels.FastLSTM,
    "Former": mlmodels.WestFormer,
    "ERT": mlmodels.WestERT,
    "GPT": mlmodels.WestGPT,
}


def _latest_checkpoint(model_dir: pathlib.Path) -> pathlib.Path:
    """Return the latest checkpoint (highest epoch number) in *model_dir*."""
    ckpts = natsort.natsorted(model_dir.glob('*.pt'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return ckpts[-1]


def _load_model(model, ckpt_path: pathlib.Path):
    """Load weights from a checkpoint file into *model*."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    screen_print(f"Loaded checkpoint: {ckpt_path.name}  (epoch {epoch})")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    proj_cfg = get_proj_config()
    base_config = proj_cfg.base_config
    model_config = load_yaml_config(args.config)
    config = deep_merge(base_config, model_config)

    model_params = config["model"]
    data_params = config["data"]
    train_params = config.get("train", {})
    base_dir = os.path.expandvars(config['base_dir'])
    random_init(seed=train_params.get('random_seed', 3407))

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

    sample_num = data_params.get('sample_num')
    if isinstance(sample_num, int):
        h5s = h5s[:sample_num]
    if args.max_shots is not None:
        h5s = h5s[:args.max_shots]
    if rank == 0:
        screen_print(f'{len(h5s)} h5 files for inference')

    if data_params.get('shuffle', False):
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

    # Resolve checkpoint
    model_name = model_params['name']
    if args.checkpoint:
        ckpt_path = pathlib.Path(args.checkpoint)
    else:
        model_dir = pathlib.Path(base_dir) / config['summary']['root_dir'] / 'Model'
        ckpt_path = _latest_checkpoint(model_dir)

    # Build and load model
    model_fn = MODEL_MAP[model_name]
    model = model_fn(**model_params)
    model = _load_model(model, ckpt_path)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        screen_print(f"Model:      {model_name}")
        screen_print(f"Checkpoint: {ckpt_path}")
        screen_print(f"Parameters: {total_params / 1e3:.0f} K")
        screen_print(f"World size: {world_size}")

    # Prediction output directory
    pred_dir = pathlib.Path(args.pred_dir) if args.pred_dir else \
        pathlib.Path(base_dir) / config['summary']['root_dir'] / 'Predictions'
    if rank == 0:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Build inference data loaders (use test split)
    data_loaders = my_data_gen.sp_ratio_wz()
    infer_loaders = data_loaders[2]  # test split

    # Prepare denormalization arrays for inference_fn
    nodes = input_nodes + output_nodes
    mean = MS_df.loc['mean', nodes].to_numpy()[-len(output_nodes):]
    stDev = MS_df.loc['stDev', nodes].to_numpy()[-len(output_nodes):]

    my_model_infer = model_dist.ModelInfer(
        infer_loaders, inference_fn,
        hat_data_dir=str(pred_dir),
        step_size=model_params['step_size'],
        window_size=model_params['window_size'],
        mean=mean,
        stDev=stDev,
    )
    my_model_infer.run_infer(model)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="WEST D0 distributed inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Model YAML config path (e.g. configs/former.yml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (default: latest in model dir)")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="Directory to write predictions (default: <root_dir>/Predictions)")
    parser.add_argument("--max-shots", type=int, default=None,
                        help="Limit number of shots (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    if os.getenv('SLURM_JOB_ID') is None and os.getenv('PBS_JOBID') is None:
        proj_cfg = get_proj_config()
        cuda_devices = proj_cfg.base_config.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    main()
