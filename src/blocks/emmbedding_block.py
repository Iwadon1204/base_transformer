#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch.nn as nn
from torch import Tensor

sys.path.append("..")

from layers.positional_encording import PostionalEncodingLayer
from layers.layer_embedding import TokenEmbeddingLayer
from util.model_setting import GlobalModelSetting
from util import const

"""
author by Iwadon
"""


class TransformerEmbeddingBlock(nn.Module):
    """
    埋め込み層本体
    入力されたトークン列
    """

    def __init__(self):
        """
        Transformerにおける埋め込みブロックを初期化します
        """
        super(TransformerEmbeddingBlock, self).__init__()

        setting = GlobalModelSetting.get_instance()

        # ドロップアウト率を取得
        drop_prob = setting.get_setting(const.KEY_DROPOUT_RATE)

        # 埋め込みレイヤーの初期化
        self.__layer_token_embedding = TokenEmbeddingLayer()
        # 位置埋め込みレイヤーの初期化
        self.__layer_positional_encoding = PostionalEncodingLayer()
        # ドロップアウト
        self.__drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: 単語ID列(Tensor) [batch * input_length]
        :return: [batch * input_length * d_model]
        """
        # 埋め込み表現を獲得 out_token_emb: [batch * max_length * d_model]
        out_token_emb = self.__layer_token_embedding(x)
        # 位置エンコーディングを取得。本処理を行うことでトークン列の位置情報を埋め込みます。out_positional_encode: [input_len * d_model]
        out_positional_encode = self.__layer_positional_encoding(x)
        # 埋め込み表現に位置エンコーディングを加算し,dropout
        return self.__drop_out(out_token_emb + out_positional_encode)
