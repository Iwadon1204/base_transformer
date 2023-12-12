#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from script.blocks.multi_head_attention import MultiHeadAttentionBlock
from script.layers.feed_forward_network import PositionwiseFeedForward
from script.layers.layer_normarize import LayerNorm


class DecoderBlock(nn.Module):
    """
    Transformer　デコーダーモデルを構成するブロック
    """
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float):
        """
        :param d_model: 時系列データの各時系列における次元数
        :param ffn_hidden: FFNレイヤーの隠れ層の数
        :param n_head: ヘッド数
        :param drop_prob: ドロップアウト率
        """
        super(DecoderBlock, self).__init__()
        # Scale dot Attentionレイヤー(self attention)
        self.layer_self_attention = MultiHeadAttentionBlock(d_model=d_model, n_head=n_head)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.layer_dropout1 = nn.Dropout(p=drop_prob)

        # Scale dot Attentionレイヤー(source-target attention)
        self.layer_src_trg_attention = MultiHeadAttentionBlock(d_model=d_model, n_head=n_head)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.layer_dropout2 = nn.Dropout(p=drop_prob)

        # FFNレイヤー
        self.layer_feedforward = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNorm(d_model=d_model)
        self.layer_dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, decoder_input: Tensor, encoder_output: Tensor, decoder_mask: Tensor,
                encoder_mask: Tensor) -> Tensor:
        """

        :param decoder_input: デコーダーへの入力 [batch * max_length * d_model]
        :param encoder_output: エンコーダーからの出力 [batch * max_length * d_model]
        :param decoder_mask: デコーダーのMASK [batch * 1 * max_length * max_length]
        :param encoder_mask: エンコーダーのマスク [batch * 1 * 1 * max_length]
        :return:
        """
        tmp_x = decoder_input
        x = self.layer_self_attention(q=decoder_input, k=decoder_input, v=decoder_input, mask=decoder_mask)

        x = self.layer_dropout1(x)
        x = self.layer_norm1(x + tmp_x)

        if encoder_output is not None:
            tmp_x = x
            x = self.layer_src_trg_attention(q=x, k=encoder_output, v=encoder_output, mask=encoder_mask)

            x = self.layer_dropout2(x)
            x = self.layer_norm2(x + tmp_x)

        tmp_x = x
        x = self.layer_feedforward(x)
        x = self.layer_dropout3(x)
        x = self.layer_norm3(x + tmp_x)
        return x
