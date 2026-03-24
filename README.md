# Machine Learning Prediction of Plasma Behavior from Discharge Configurations on WEST

This repository contains the code for the paper:

> **Machine learning prediction of plasma behavior from discharge configurations on WEST**
>
> Chenguang Wan, Feda Almuhisen\*, Philippe Moreau, Remy Nouailletas, Zhisong Qu\*, Youngwoo Cho, Robin Varennes, Kyungtak Lim, Kunpeng Li, Jia Huang, Weidong Chen, Jiangang Li, Xavier Garbet and WEST Team
>
> Published 3 March 2026, DOI: [10.1088/1741-4326/ae4917](https://doi.org/10.1088/1741-4326/ae4917)

## Overview

A transformer-based machine learning model for predicting key global plasma parameters on the Tungsten (W) Environment in Steady-State Tokamak (WEST), including:

- Normalized beta ($\beta_n$)
- Toroidal beta ($\beta_t$)
- Poloidal beta ($\beta_p$)
- Plasma stored energy ($W_{mhd}$)
- Safety factor at the magnetic axis ($q_0$)
- Safety factor at the 95% flux surface ($q_{95}$)

The model uses only pre-discharge signals (magnetic coil currents, auxiliary heating power, plasma current reference, and line-averaged plasma density) to predict plasma behavior. Trained on 550 discharges from WEST campaigns, it achieves an average MSE loss of 0.026, an average $R^2$ of 0.94, and inference times on the order of 0.1 s.

## Project Structure

```
WestD0Clean/
├── configs/
│   ├── base.yml            # Shared configuration (nodes, data paths, data params)
│   └── former.yml          # Model-specific config (architecture, training, optimizer)
├── src/
│   ├── data_gen.py         # Dataset classes and data loaders (StdWESTShotDS, H5GenDataLoader)
│   ├── data_utils.py       # Data utility functions
│   ├── mlmodels.py         # Model definitions (WestFormer, WestERT, WestGPT, FastLSTM, MLP)
│   ├── model_dist.py       # Distributed training/inference engines (torchrun-compatible)
│   ├── proj_config.py      # Singleton project configuration (paths derived from database_dir)
│   └── utils.py            # Loss functions, IO, and common utilities
├── run_data_pre.py         # Data preprocessing pipeline (subcommand-based CLI)
├── run_train.py            # Distributed training entry point
└── run_infer.py            # Distributed inference entry point
```

## Configuration

Configs follow a **base + model override** pattern. `configs/base.yml` defines shared settings (node definitions, data paths, data parameters). Model configs (e.g. `configs/former.yml`) override or extend specific fields via deep merge.

## Usage

### Data Preprocessing

```bash
python run_data_pre.py <subcommand> [--num-workers N]
```

### Training

Single GPU:
```bash
python run_train.py --config configs/former.yml
```

Multi-GPU with torchrun:
```bash
torchrun --nproc_per_node=4 run_train.py --config configs/former.yml
```

Restore from checkpoint:
```bash
torchrun --nproc_per_node=4 run_train.py --config configs/former.yml --train-type restore
```

### Inference

```bash
torchrun --nproc_per_node=4 run_infer.py --config configs/former.yml
```

With a specific checkpoint:
```bash
python run_infer.py --config configs/former.yml --checkpoint path/to/model.pt
```

## Requirements

- Python 3.11+
- PyTorch (with CUDA support)
- pandas, numpy, h5py, natsort, tqdm, PyYAML, tensorboard
