# -*- coding: utf-8 -*-
import argparse
import pathlib
import torch
import os

from src import data_gen
from src import mlmodel, model_dist
from private_modules import load_yaml_config, clean_dir, screen_print
from private_modules.Torch import tools, qkmodels
from ray import tune
import pandas as pd
import numpy as np
import time
import random
from functools import partial

from v2.src import data_utils

# add more length mask in the future. 
@tools.ray_start
def main_run(config, num_samples):
    train_params = config["train"]
    model_params = config["model"]
    data_params = config["data"]
    base_dir = config['base_dir']
    base_dir = os.path.expandvars(base_dir)
    tools.random_init(seed=train_params['random_seed'])

    data_dir = data_params['data_dir']
    data_dir = os.path.expandvars(data_dir)
    data_dir = pathlib.Path(data_dir)

    # stat_f = data_dir.joinpath('h5_stat.csv')
    stat_f = pathlib.Path("./Database/Stat/h5_stat.csv")
    stat_df = pd.read_csv(stat_f, index_col=0)

    h5s = list(data_dir.iterdir())
    world_size = torch.cuda.device_count()

    MS_file = stat_f.parent.joinpath('h5_global_MS.csv')
    MS_df = pd.read_csv(MS_file, index_col=0)

    h5s = list(data_dir.iterdir())

    world_size = torch.cuda.device_count()  # for single node.

    # MS_file = data_dir.joinpath('h5_global_MS.csv')
    MS_file = data_dir.joinpath('h5_global_MS_add.csv')
    MS_df = pd.read_csv(MS_file, index_col=0)

    input_list = data_params['input_list']
    output_list = data_params['output_list']
    nodes = []
    input_nodes, output_nodes = [], []
    for dummy_list_name in input_list:
        if "_real" in dummy_list_name: 
            dummy_list_name = dummy_list_name[:-5]
            i = 3
        if "_ref" in dummy_list_name: 
            dummy_list_name = dummy_list_name[:-4]
            i = 0
        input_nodes.extend([f"{dummy_node}_{i}" for dummy_node in config['nodes'][dummy_list_name]])
    for dummy_list_name in output_list:
        output_nodes.extend(config['nodes'][dummy_list_name])
    nodes.extend(input_nodes)
    nodes.extend(output_nodes)
    sample_num = data_params['sample_num']
    if type(sample_num) == int:
        h5s = h5s[:sample_num]
    print(f'{len(h5s)} h5 files input')

    shuffle = data_params['shuffle']
    if shuffle:
        random.shuffle(h5s)
    my_data_gen = data_gen.H5GenDataLoader(
        h5s=h5s,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        batch_size=data_params['batch_size'] * world_size,
        num_workers=data_params['num_workers'],
        DS=data_gen.StdMCFShotWinDS,
        MS_df=MS_df,
        stat_df=stat_df,
        pin_memory=True,
        world_size=world_size,
    )
    data_loaders = my_data_gen.sp_ratio_wz(split_ratio=[0.99, 0.01, 0.001])
    tra_loaders, val_loaders, test_loaders = data_loaders[0], data_loaders[1], data_loaders[2]
    # train_steps_per_epoch = len(tra_loader) + batch_size - 1

    model_params = config['model']
    num_epochs = train_params['num_epochs']
    # optimer_fn = torch.optim.SGD
    scheduler_fn = eval(config['optimizer']['scheduler'])
    train_params['optimer_fn'] = eval(config['optimizer']['name'])
    # This is a scaling law, lr / lr_0 = batch_size / batch_size_0
    train_params['learning_rate'] = float(
        config['optimizer']['lr']) * world_size
    train_params['scheduler_fn'] = scheduler_fn
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
        # "RNN": {"train": mlmodels.RNN_TransSS,
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
        "MLP": {"train": qkmodels.MLP,
                'build_model': build_model_MLP,
                # "loss_fn": utils.calc_loss_MLP,
                "loss_fn": data_utils.calc_loss_MLP,
                "search_space": {
                    'num_layers': tune.randint(1, 5),
                    # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
                    "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
                }},
        "FastLSTM": {"train": mlmodels.FastLSTM,
                     'build_model': build_model_RNN,
                     "loss_fn": data_utils.calc_loss_MLP,
                     #   "loss_fn": utils.test_output,
                     "search_space": {
                         'num_layers': tune.randint(1, 5),
                         # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
                         "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
                     }},
        "Former": {"train": mlmodels.WestFormer,
                   "build_model": build_model_Former,
                   "loss_fn": partial(data_utils.calc_loss_Former, **train_params,),
                   "search_space": {
                       'num_layers': tune.qrandint(1, 8, 2),
                       # 'num_layers': tune.sample_from(lambda spec: 2 ** spec.config.uniform),
                       "learning_rate": tune.qloguniform(1e-4, 1e-1, 5e-5),
                   },
                   },
        # "ERT": {"train": mlmodels.RZIpERT,
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
    my_model_tune = model_tune.ModelTuneRNN(tra_loaders,
                                            val_loaders,
                                            num_epochs,
                                            world_size,
                                            loss_fn,
                                            train_base_dir,
                                            train_params,)
    start = time.time()
    if not train_params['is_train']:
        search_space = model_pairs['search_space']
        build_model = model_pairs['build_model']
        my_model_tune.run_tune(num_samples=num_samples,
                               tune_config=None,
                               build_model=build_model,
                               search_space=search_space,
                               )
    else:
        my_model = model_pairs['train']
        model = my_model(**model_params,)
        my_model_tune.run_train(
            model, 
            restore=train_params['restore'],
        )

    end = time.time()
    screen_print(f"Running time {end - start} senconds")