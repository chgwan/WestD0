# -*- coding: utf-8 -*- 
import torch
from torch import nn
from torch import Tensor
import random
from typing import Callable, Optional
import math
from private_modules.Torch import qkmodels
from Models import Swin

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

class EmbedPostion(nn.Sequential):
    def __init__(
            self,
            input_dim,
            embed_dim,
            window_size,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = nn.Sequential(
            # Make Channel last to channel first
            qkmodels.Permute([0, 2, 1]),
            # Conv1d input: [B, C_{in}, L_{in}]
            # Conv1d output: [B, C_{out}, L_{out}]
            # modify the channel of input.
            nn.Conv1d(input_dim, embed_dim, kernel_size=1, stride=1),
            # Make channel last to channel first
            qkmodels.Permute([0, 2, 1]),
            nn.LayerNorm(embed_dim),
            qkmodels.PositionalEncoding(embed_dim, dropout, max_len=window_size),
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
    ) -> None:
        super().__init__()
        self.noise_ratio = noise_ratio
        self.enc_embed = EmbedPostion(input_dim,
                                      embed_dim,
                                      window_size=window_size,)
        # self.dec_embed = EmbedPostion(output_dim,
        #                               embed_dim,
        #                               window_size=window_size,)
        self.trans = nn.Transformer(
            d_model=embed_dim,
            num_encoder_layers=num_layers // 2,
            num_decoder_layers=num_layers // 2,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.restore_layer = nn.Linear(
            in_features=embed_dim,
            out_features=output_dim,)

    def forward(self, enc_inputs):
        # if self.training:
        #     # Input add noise
        #     noise = torch.randn_like(enc_inputs) * self.noise_ratio
        #     enc_inputs = ((noise + enc_inputs).detach() -
        #                   enc_inputs).detach() + enc_inputs
        #     # one-step ahead noise.
        #     noise = torch.randn_like(dec_inputs) * self.noise_ratio
        #     dec_inputs = ((noise + dec_inputs).detach() -
        #                   dec_inputs).detach() + dec_inputs
        # enc_inputs = self.enc_embed(enc_inputs)
        # dec_inputs = self.dec_embed(dec_inputs)
        enc_inputs = self.enc_embed(enc_inputs)
        maxlen = enc_inputs.size(1)
        causal_mask = self.trans.generate_square_subsequent_mask(
            maxlen,
            device=enc_inputs.device,)
        # dec_outputs = self.trans(src=enc_inputs,
        #                          tgt=,
        #                         #  src_mask=causal_mask,
        #                         #  tgt_mask=causal_mask,
        #                          src_is_causal=True,
        #                          tgt_is_causal=True,)
        enc_outputs = self.trans.encoder(enc_inputs, causal_mask)
        dec_outputs = self.trans.decoder(enc_inputs, enc_outputs, causal_mask)
        Y_hat = self.restore_layer(dec_outputs)
        return Y_hat

    def _generate_padding_mask(self, maxlen, valid_len):
        # 新的 维度添加 [None, :] 在第0维添加新的维度
        mask = torch.arange((maxlen),
                            dtype=torch.int,
                            device=self.device)[None, :] < valid_len[:, None]
        return ~mask

    # @torch.no_grad()
    # def inference(self, enc_inputs):
    #     """
    #     Args:
    #         enc_inputs (Tensor): [batch_size, seq_len, input_size],
    #             encoder input of the network
    #         dec_inputs (Tensor): [batch_size, 1, output_size],
    #             decoder input of the network.
    #         state (Tensor): RNN states
    #     Returns: model_outputs
    #         * model_ouputs (Tensor): [batch_size, seq_len, output_size]
    #     """
    #     # TODO(Mr.wan): improve to compatible with WestD0 data
    #     enc_l = enc_inputs.shape[1]
    #     enc_inputs = self.enc_embed(enc_inputs)
    #     src_mask = self.trans.generate_square_subsequent_mask(
    #         enc_l, 
    #         device=enc_inputs.device,
    #         )
    #     enc_outputs = self.trans.encoder(enc_inputs, src_mask)
    #     # memory = enc_outputs[:, -1, :]

        
    #     for _ in range(enc_l):
    #         tgt_mask = self.trans.generate_square_subsequent_mask(
    #             dec_input.size(1),
    #             device=dec_input.device)
    #         model_input = self.dec_embed(dec_input)
    #         model_output = self.trans.decoder(
    #             model_input, 
    #             memory=enc_outputs, 
    #             tgt_mask=tgt_mask)
    #         model_output = self.restore_layer(model_output)
    #         # print(model_output.size())
    #         dec_input = torch.cat([dec_input, model_output[:, -1:, :]], dim=1)
    #     # remove the start
    #     dec_input = dec_input[:, 1:, :]
    #     return dec_input

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
    ) -> None:
        super().__init__()
        self.noise_ratio = noise_ratio
        self.enc_embed = EmbedPostion(input_dim,
                                      embed_dim,
                                      window_size=window_size,)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.ERT = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers)
        self.dense = nn.Linear(
            in_features=embed_dim,
            out_features=output_dim,)

if __name__ == "__main__":
    batch_size = 1
    seq_len = int(10 ** 4)
    input_dim = 19
    output_dim = 6
    embed_dim = 512

    input_seq = torch.zeros(batch_size, seq_len, input_dim)
    tgt_seq = torch.zeros(batch_size, seq_len, output_dim)


    model = WestFormer(
        input_dim,
        embed_dim,
        output_dim,
        window_size=int(5 * 10 ** 4))
    
    model.cuda()
    input_seq = input_seq.cuda()

    
    output_seq = model(input_seq)
    print(output_seq.shape)