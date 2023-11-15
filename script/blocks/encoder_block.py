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


class EncoderBlock(nn.Module):
    """
    TransformerにおけるEncoderブロック
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderBlock, self).__init__()
        # MultiHead-Attention
        self.attention = MultiHeadAttentionBlock(d_model=d_model, n_head=n_head, dropout_rate=drop_prob)
        # Normalize1
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # FeedForwardNetwork
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # Normalize2
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, encoder_input: Tensor, s_mask: Tensor):
        """

        :param encoder_input: [batch * max_length * d_model]
        :param s_mask: [batch * 1 * 1 * max_length]
        :return: [batch * max_length * d_model]
        """
        _x = encoder_input
        x = self.attention(q=encoder_input, k=encoder_input, v=encoder_input, mask=s_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x