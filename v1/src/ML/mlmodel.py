# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import Tensor
import random
from typing import Callable, Optional
import math

from private_modules.Torch import qkmodels

class FastLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        output_dim: int = 1,
        noise_ratio: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=output_dim,
            dropout=dropout_rate,
            )
    def forward(self, X, state=None):
        Y, state = self.model(X, state)
        return Y