#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from blocks.multi_head_attention import MultiHeadAttentionBlock
from layers.feed_forward_network import PositionwiseFeedForward
from layers.layer_normarize import LayerNorm
from util.model_setting import GlobalModelSetting
from util import const


class DecoderBlock(nn.Module):
    """
    Transformer　デコーダーモデルを構成するブロック
    """
    def __init__(self):
        super(DecoderBlock, self).__init__()

        setting = GlobalModelSetting.get_instance()

        # ドロップアウト率を取得
        drop_prob = setting.get_setting(const.KEY_DROPOUT_RATE)

        # Scale dot Attentionレイヤー(self attention)
        self.__layer_self_attention = MultiHeadAttentionBlock()
        self.__layer_norm1 = LayerNorm()
        self.__layer_dropout1 = nn.Dropout(p=drop_prob)

        # Scale dot Attentionレイヤー(source-target attention)
        self.__layer_src_trg_attention = MultiHeadAttentionBlock()
        self.__layer_norm2 = LayerNorm()
        self.__layer_dropout2 = nn.Dropout(p=drop_prob)

        # FFNレイヤー
        self.__layer_feedforward = PositionwiseFeedForward()
        self.__layer_norm3 = LayerNorm()
        self.__layer_dropout3 = nn.Dropout(p=drop_prob)

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
        x = self.__layer_self_attention(q=decoder_input, k=decoder_input, v=decoder_input, mask=decoder_mask)

        x = self.__layer_dropout1(x)
        x = self.__layer_norm1(x + tmp_x)

        if encoder_output is not None:
            tmp_x = x
            x = self.__layer_src_trg_attention(q=x, k=encoder_output, v=encoder_output, mask=encoder_mask)

            x = self.__layer_dropout2(x)
            x = self.__layer_norm2(x + tmp_x)

        tmp_x = x
        x = self.__layer_feedforward(x)
        x = self.__layer_dropout3(x)
        x = self.__layer_norm3(x + tmp_x)
        return x
