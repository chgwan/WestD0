# -*- coding: utf-8 -*-
import pandas as pd
import pathlib
import numpy as np
# from scipy.io import 
import os
from src.ML import utils, data_gen, mlmodel
# from matplotlib import pyplot as plt
from private_modules import load_yaml_config
from private_modules.Torch import MCFDS
import torch.utils.data as Data
from matplotlib import pyplot as plt

import torch

Database_dir = '/home/chenguang.wan/Papers/DataTest/Database'
Database_dir = pathlib.Path(Database_dir)
mat_dir = os.path.expandvars("$DATABASE_PATH/DataBase/WEST/PCS")
mat_dir = pathlib.Path(mat_dir)

config_file = '/home/chenguang.wan/Papers/DataTest/Database/configs/data_config.yml'
config = load_yaml_config(config_file)

data_params = config['data']
model_params = config['model']
node_maps = config['nodes']
input_nodes, output_nodes = [], []

input_list = data_params['input_list']
output_list = data_params['output_list']

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

ms_file = Database_dir.joinpath("Stat/node_MS.csv")
MS_df = pd.read_csv(ms_file, index_col=0)



# model_path = "/home/chenguang.wan/Papers/DataTest/Database/FLSTM/Model/1-0.019852-0.020183.pt"
model_path = "/home/chenguang.wan/Papers/DataTest/Database/FLSTM/Model/3-0.014479-0.012853.pt"
model_dict = torch.load(model_path, weights_only=True, map_location='cpu')

model = mlmodel.FastLSTM(**model_params)
model.load_state_dict(model_dict)
model.eval()

shot = 58308
mat_file = mat_dir.joinpath(f"DCS_archive_{shot}.mat")
mat_files = [mat_file]

ds = data_gen.StdWESTShotDS(
    mat_files, input_nodes, output_nodes,
    MS_df = MS_df,
)
data = next(iter(ds))
X, Y_tgt = data[0], data[1]
X = X[np.newaxis, :]

X = torch.tensor(X).float()
with torch.no_grad():
    Y_hat = model(X)
Y_hat = Y_hat.numpy()

Y_hat = np.squeeze(Y_hat)
Y_tgt = np.squeeze(Y_tgt)

# fig, axes = plt.subplots(2, 1)
# axes[0].plot(Y_hat, 'inference')
# axes[1].plot(Y_tgt, 'target')
# axes[0].legend()
# axes[1].legend()
# fig.savefig('a.png')
plt.close('all')
plt.plot(Y_tgt)
plt.plot(Y_hat)
plt.savefig('a.png')
print(Y_hat)