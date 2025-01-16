# -*- coding: utf-8 -*-
import pathlib
import numpy as np
from src.DataPre import stat_nodes
import argparse

Database_dir = './Database'
mat_data_dir = '/donnees/NTU/NTU'
Database_dir = pathlib.Path(Database_dir)
mat_data_dir = pathlib.Path(mat_data_dir)


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="Run function")
    parser.add_argument(
        "--run-code",
        type=str,
        help="The function you want to run",
        required=True,
    )
    args = parser.parse_args()
    return args


def stat_dir():
    config_path = './config/data_config.yml'
    file_list = list(mat_data_dir.glob("DCS_archive_*.mat"))
    file_list = file_list[:]
    config = stat_nodes.load_yaml_config(config_path)
    nodes = config["nodes"]
    
    stat_file = Database_dir.joinpath('Stat/node_stat.csv')
    stat_nodes.scan_dir(file_list, nodes, stat_file)
    calc_nodes = []
    for node in nodes:
        calc_nodes.append(f"{node}_0")
        calc_nodes.append(f"{node}_3")
    MS_file = Database_dir.joinpath('Stat/node_MS.csv')
    stat_nodes.calMS(calc_nodes, stat_file, MS_file=MS_file)

if __name__ == "__main__":
    args = parse_args()
    run_code = args.run_code.strip()
    run_code = f"{run_code}"
    eval(run_code)