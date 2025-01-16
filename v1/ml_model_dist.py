# -*- coding: utf-8 -*-
import argparse
import pathlib
import torch
import os

from src.ML import data_gen
from src.ML import mlmodel, model_dist, utils
from private_modules import load_yaml_config, clean_dir, screen_print
from private_modules.Torch import tools, qkmodels
from ray import tune
import pandas as pd
import numpy as np
import time
import random
from functools import partial

# @tools.ray_start

def main_run(config):
    train_params = config["train"]
    model_params = config["model"]
    data_params = config["data"]
    base_dir = config['base_dir']
    base_dir = os.path.expandvars(base_dir)
    tools.random_init(seed=train_params['random_seed'])

    data_dir = data_params['data_dir']
    data_dir = os.path.expandvars(data_dir)
    data_dir = pathlib.Path(data_dir)
    base_dir = pathlib.Path(base_dir)

    stat_f = base_dir.joinpath("Database/Stat/node_stat.csv")
    shots = utils.get_shots(stat_f)

    mat_files = [data_dir.joinpath(f'DCS_archive_{shot}.mat') for shot in shots]
    random.shuffle(mat_files)

    world_size = torch.cuda.device_count()  # for single node.

    # MS_file = data_dir.joinpath('h5_global_MS.csv')
    # MS_file = data_dir.joinpath('h5_global_MS_add.csv')
    MS_f = base_dir.joinpath('Database/Stat/node_MS.csv')
    MS_df = pd.read_csv(MS_f, index_col=0)

    node_maps = config['nodes']
    input_list = data_params['input_list']
    output_list = data_params['output_list']
    input_nodes, output_nodes = [], []
    for nodeList_name in input_list:
        if "_real" in nodeList_name:
            dummy_list = node_maps[nodeList_name[:-5]]
            dummy_list = [f"{node}_3" for node in dummy_list]
            input_nodes.extend(dummy_list)
        elif "_ref" in nodeList_name:
            dummy_list = node_maps[nodeList_name[:-4]]
            dummy_list = [f"{node}_0" for node in dummy_list]
            input_nodes.extend(dummy_list)

    for nodeList_name in output_list:
        if "_real" in nodeList_name:
            dummy_list = node_maps[nodeList_name[:-5]]
            dummy_list = [f"{node}_3" for node in dummy_list]
            output_nodes.extend(dummy_list)
        elif "_ref" in nodeList_name:
            dummy_list = node_maps[nodeList_name[:-4]]
            dummy_list = [f"{node}_0" for node in dummy_list]
            output_nodes.extend(dummy_list)

    nodes = []
    nodes.extend(input_nodes)
    nodes.extend(output_nodes)
    sample_num = data_params['sample_num']
    if sample_num == 'None':
        sample_num = None
    if sample_num is not None:
        mat_files = mat_files[:sample_num]
    print(f'{len(mat_files)} h5 files input, with {world_size} gpus')

    shuffle = data_params['shuffle']
    if shuffle:
        random.shuffle(mat_files)
    my_data_gen = data_gen.H5GenDataLoader(
        h5s=mat_files,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        DS=data_gen.StdWESTShotDS,
        batch_size=data_params['batch_size'] * world_size,
        num_workers=data_params['num_workers'],
        MS_df=MS_df,
        shot_start=0,
        pin_memory=True,
        world_size=world_size,
    )
    my_data_gen.set_split_ratio(split_ratio=[0.80, 0.1, 0.1])
    # data_loaders = my_data_gen.sp_ratio_wz(split_ratio=[0.99, 0.01, 0.001])
    # tra_loaders, val_loaders, test_loaders = data_loaders[0], data_loaders[1], data_loaders[2]
    # train_steps_per_epoch = len(tra_loader) + batch_size - 1
    data_loaders = my_data_gen.sp_ratio_wz()
    model_params = config['model']
    num_epochs = train_params['num_epochs']
    # optimer_fn = torch.optim.SGD
    # scheduler_fn = eval(config['optimizer']['scheduler'])

    # scheduler_fn = partial(torch.optim.lr_scheduler.OneCycleLR, )

    train_params['optimer_fn'] = eval(config['optimizer']['name'])
    # This is a scaling law, lr / lr_0 = batch_size / batch_size_0
    train_params['learning_rate'] = float(
        config['optimizer']['lr']) * world_size
    # scheduler_fn = partial(torch.optim.lr_scheduler.OneCycleLR, 
    #                     max_lr=train_params['learning_rate'],
    #                     epochs=num_epochs,
    #                     steps_per_epoch=len(data_loaders[0][0]))
    train_params['optimer_fn'] = partial(torch.optim.RMSprop, momentum=0.1)
    train_params['scheduler_fn'] = None
    train_params.update(config['summary'])
    train_base_dir = os.path.join(base_dir, config['summary']['root_dir'])
    pathlib.Path(train_base_dir).mkdir(exist_ok=True)
    if not train_params['restore']:
        clean_dir(train_base_dir)
    else:
        train_params['checkpoint_path'] = os.path.join(
            base_dir,
            train_params['checkpoint_path'],
        )
    model_pair_dict = {
        # "RNN": {"train": mlmodel.RNN_TransSS,
        #         'build_model': build_model_RNN,
        #         "loss_fn": partial(utils.calc_loss_RNN_sstf,
        #                            **train_params,),
        #         "search_space": {
        #             'num_layers': tune.randint(1, 5),
        #             # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
        #             "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
        #             'hidden_size_base': tune.randint(0, 5),
        #         }
        #         },
        # "MLP": {"train": qkmodels.MLP,
        #         'build_model': build_model_MLP,
        #         # "loss_fn": utils.calc_loss_MLP,
        #         "loss_fn": utils.calc_loss_MLP,
        #         "search_space": {
        #             'num_layers': tune.randint(1, 5),
        #             # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
        #             "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
        #         }},
        "FastLSTM": {"train": mlmodel.FastLSTM,
                     'build_model': build_model_RNN,
                     "loss_fn": utils.calc_loss_MLP,
                     #   "loss_fn": utils.test_output,
                     "search_space": {
                         'num_layers': tune.randint(1, 5),
                         # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
                         "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
                     }},
        # "Former": {"train": mlmodel.RZIpFormer,
        #            "build_model": build_model_Former,
        #            "loss_fn": partial(utils.calc_loss_Former, **train_params,),
        #            "search_space": {
        #                'num_layers': tune.qrandint(1, 8, 2),
        #                # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
        #                "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
        #            },
        #            },
        # "ERT": {"train": mlmodel.RZIpERT,
        #         "build_model": build_model_ERT,
        #         "loss_fn": partial(utils.calc_loss_RNN_sstf, **train_params,),
        #         "search_space": {
        #             'num_layers': tune.qrandint(2, 10, 1),
        #             # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
        #             "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
        #         },
        #         },
    }
    model_name = model_params['name']
    model_pairs = model_pair_dict[model_name]
    loss_fn = model_pairs['loss_fn']
    start = time.time()
    my_model_train = model_dist.ModelTrainRNN(
        # tra_loaders,
        # val_loaders,
        my_data_gen,
        num_epochs,
        world_size,
        loss_fn,
        train_base_dir,
        train_params,)
    # model = torch.nn.Linear(in_features=61, out_features=76)
    model_fn = model_pairs['train']
    model = model_fn(**model_params,)
    my_model_train.run_train(model)
    end = time.time()
    screen_print(f"Running time {end - start} senconds")


# def build_model_MLP(search_space):
#     model_params = config['model']
#     input_dim = model_params['input_dim']
#     output_dim = model_params['output_dim']
#     dropout_rate = model_params['dropout_rate']
#     num_layers = search_space['num_layers']
#     base_dim = 128
#     tmp_size = []
#     for i in range(num_layers):
#         tmp_size.append(2 ** i * base_dim)
#     hidden_sizes = []
#     hidden_sizes.extend(tmp_size)
#     hidden_sizes.extend(reversed(tmp_size))
#     model = mlmodel.MLP(input_dim=input_dim,
#                         hidden_sizes=hidden_sizes,
#                         dropout_rate=dropout_rate,
#                         output_dim=output_dim)
#     return model


def build_model_RNN(search_space):
    model_params = config['model']
    input_dim = model_params['input_dim']
    output_dim = model_params['output_dim']
    dropout_rate = model_params['dropout_rate']

    num_layers = search_space['num_layers']

    hidden_size_base = search_space['hidden_size_base']
    base_dim = 5
    hidden_size = 2 ** (hidden_size_base + base_dim)

    model = mlmodel.FastLSTM(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=output_dim)
    return model


# def build_model_Former(search_space):
#     model_params = config['model']
#     input_dim = model_params['input_dim']
#     embed_dim = model_params['embed_dim']
#     output_dim = model_params['output_dim']
#     dropout_rate = model_params['dropout_rate']
#     noise_ratio = model_params['noise_ratio']
#     window_size = model_params['window_size']

#     num_layers = search_space['num_layers']

#     model = mlmodel.RZIpFormer(
#         input_dim=input_dim,
#         embed_dim=embed_dim,
#         output_dim=output_dim,
#         window_size=window_size,
#         num_layers=num_layers,
#         dropout_rate=dropout_rate,
#         noise_ratio=noise_ratio,)
#     return model


# def build_model_ERT(search_space):
#     model_params = config['model']
#     input_dim = model_params['input_dim']
#     embed_dim = model_params['embed_dim']
#     output_dim = model_params['output_dim']
#     dropout_rate = model_params['dropout_rate']
#     noise_ratio = model_params['noise_ratio']
#     window_size = model_params['window_size']

#     num_layers = search_space['num_layers']

#     model = mlmodel.RZIpERT(
#         input_dim=input_dim,
#         embed_dim=embed_dim,
#         output_dim=output_dim,
#         window_size=window_size,
#         num_layers=num_layers,
#         dropout_rate=dropout_rate,
#         noise_ratio=noise_ratio,)
#     return model


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="run gx and gkw")
    parser.add_argument(
        "--config",
        type=str,
        help="The function you want to run",
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    if os.getenv('SLURM_JOB_ID') is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
    main_run(config)
