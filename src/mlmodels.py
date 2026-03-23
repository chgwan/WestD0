# -*- coding: utf-8 -*-
import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=None, output_dim=None,
                 activation_layer=nn.GELU, dropout_rate=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []
        layer_sizes = [input_dim] + list(hidden_sizes)
        self.layers = nn.ModuleList()
        drop = nn.Dropout(dropout_rate)
        act = activation_layer()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(act)
            self.layers.append(drop)
        self.layers.append(nn.Linear(layer_sizes[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers,
                 dropout_rate=0.0, output_dim=1, bidirectional=False, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.mlp = nn.Linear(hidden_size * (bidirectional + 1), output_dim)

    def forward(self, X, state=None):
        Y, state = self.lstm(X, state)
        Y = self.mlp(Y)
        return Y, state


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, embed_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class EmbedPosition(nn.Sequential):
    def __init__(self, input_dim, embed_dim, window_size, dropout=0.1):
        layers = nn.Sequential(
            Permute([0, 2, 1]),
            nn.Conv1d(input_dim, embed_dim, kernel_size=1, stride=1),
            Permute([0, 2, 1]),
            nn.LayerNorm(embed_dim),
            PositionalEncoding(embed_dim, dropout, max_len=window_size),
        )
        super().__init__(*layers)


class WestFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        window_size: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        noise_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.noise_ratio = noise_ratio
        self.enc_embed = EmbedPosition(input_dim, embed_dim, window_size=window_size)
        self.trans = nn.Transformer(
            d_model=embed_dim,
            num_encoder_layers=num_layers // 2,
            num_decoder_layers=num_layers // 2,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.restore_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, enc_inputs):
        enc_inputs = self.enc_embed(enc_inputs)
        maxlen = enc_inputs.size(1)
        causal_mask = self.trans.generate_square_subsequent_mask(
            maxlen, device=enc_inputs.device
        )
        enc_outputs = self.trans.encoder(enc_inputs, causal_mask)
        dec_outputs = self.trans.decoder(enc_inputs, enc_outputs, causal_mask)
        return self.restore_layer(dec_outputs)


class WestERT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        window_size: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        noise_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.noise_ratio = noise_ratio
        self.enc_embed = EmbedPosition(input_dim, embed_dim, window_size=window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.former = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.restore_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, enc_inputs):
        enc_inputs = self.enc_embed(enc_inputs)
        maxlen = enc_inputs.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            maxlen, device=enc_inputs.device
        )
        enc_outputs = self.former(enc_inputs, causal_mask, is_causal=True)
        return self.restore_layer(enc_outputs)


class WestGPT(WestERT):
    def __init__(self, input_dim, embed_dim, output_dim, window_size=1024,
                 num_heads=8, num_layers=2, dropout_rate=0.1, noise_ratio=0.1, **kwargs):
        super().__init__(input_dim, embed_dim, output_dim, window_size,
                         num_heads, num_layers, dropout_rate, noise_ratio, **kwargs)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.former = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, enc_inputs):
        enc_inputs = self.enc_embed(enc_inputs)
        maxlen = enc_inputs.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            maxlen, device=enc_inputs.device
        )
        enc_outputs = self.former(
            enc_inputs, enc_inputs,
            causal_mask, causal_mask,
            tgt_is_causal=True, memory_is_causal=True,
        )
        return self.restore_layer(enc_outputs)
