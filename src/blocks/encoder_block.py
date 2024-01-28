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

from util.model_setting import GlobalModelSetting
from util import const


class EncoderBlock(nn.Module):
    """
    TransformerにおけるEncoderブロック
    """
    def __init__(self):
        super(EncoderBlock, self).__init__()

        setting = GlobalModelSetting.get_instance()

        # ドロップアウト率を取得
        drop_prob = setting.get_setting(const.KEY_DROPOUT_RATE)

        # MultiHead-Attention
        self.__attention_block = MultiHeadAttentionBlock()
        # Normalize1
        self.__norm1 = LayerNorm()
        self.__dropout1 = nn.Dropout(p=drop_prob)
        # FeedForwardNetwork
        self.__ffn = PositionwiseFeedForward()
        # Normalize2
        self.__norm2 = LayerNorm()
        self.__dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, encoder_input: Tensor, mask: Tensor):
        """
        :param encoder_input: エンコーダーへの入力 [batch * max_length * d_model]
        :param mask: マスク行列  [batch * 1 * 1 * max_length]
        :return: [batch * max_length * d_model]
        """
        tmp_x = encoder_input
        x = self.__attention_block(q=encoder_input, k=encoder_input, v=encoder_input, mask=mask)

        x = self.__dropout1(x)
        x = self.__norm1(x + tmp_x)

        tmp_x = x
        x = self.__ffn(x)

        x = self.__dropout2(x)
        x = self.__norm2(x + tmp_x)
        return x