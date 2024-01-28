#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from blocks.emmbedding_block import TransformerEmbeddingBlock
from blocks.encoder_block import EncoderBlock

from util.model_setting import GlobalModelSetting
from util import const


class Encoder(nn.Module):
    """
    エンコーダーモデル
    """
    def __init__(self):
        super(Encoder, self).__init__()
        setting = GlobalModelSetting.get_instance()
        # エンコーダーのレイヤー数を取得
        enc_layer_num = setting.get_setting(const.KEY_ENC_LAYER_NUM)
        # 埋め込みブロック
        self.__emb = TransformerEmbeddingBlock()
        # Encoderブロックをレイヤーの数だけ作成
        self.__layers = nn.ModuleList([EncoderBlock() for _ in range(enc_layer_num)])

    def forward(self, x: Tensor, encoder_mask: Tensor) -> Tensor:
        """
        :param x: 入力トークン列(Tensor) [batch * max_length]
        :param encoder_mask: マスク(Tensor) [batch * 1 * 1 * max_length]
        :return: Encoderモデル出力(Tensor) [batch * max_length * d_model]
        """
        # 埋め込み層によって獲得されるTensor[batch * max_length * d_model]
        x = self.__emb(x)

        for layer in self.__layers:
            x = layer(x, encoder_mask)
        return x
