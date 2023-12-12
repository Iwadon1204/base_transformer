#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import sys
import torch.nn as nn
from torch import Tensor

sys.path.append("..")

from blocks.multi_head_attention import MultiHeadAttentionBlock
from layers.feed_forward_network import PositionwiseFeedForward
from layers.layer_normarize import LayerNorm


class EncoderBlock(nn.Module):
    """
    TransformerにおけるEncoderブロック
    """
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float):
        """
        :param d_model: モデル次元数
        :param ffn_hidden: FFNレイヤーの隠れ層
        :param n_head: ヘッド数
        :param drop_prob: ドロップアウト率
        """
        super(EncoderBlock, self).__init__()
        # MultiHead-Attention
        self.attention_block = MultiHeadAttentionBlock(d_model=d_model, n_head=n_head)
        # Normalize1
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        # FeedForwardNetwork
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # Normalize2
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, encoder_input: Tensor, mask: Tensor):
        """
        :param encoder_input: エンコーダーへの入力 [batch * max_length * d_model]
        :param mask: マスク行列  [batch * 1 * 1 * max_length]
        :return: [batch * max_length * d_model]
        """
        _x = encoder_input
        x = self.attention_block(q=encoder_input, k=encoder_input, v=encoder_input, mask=mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x