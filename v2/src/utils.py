# -*- coding: utf-8 -*-
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import torch
from private_modules.Torch import tools

def calc_loss_MLP(
    model,
    X: torch.tensor,
    Y: torch.tensor,
    Y_len,
    batch_output_flags,
    current_steps = None,
    **kwargs,
):
    loss_fn = tools.MaskedMSELoss(reduction='mean')
    Y_hat = model(X)
    loss = loss_fn(Y_hat, Y, None, None)
    # print(loss)
    # print(X.shape)
    # print(Y)
    if torch.isnan(loss).any().item():
        raise ValueError(f"Nan in loss")
    return loss

