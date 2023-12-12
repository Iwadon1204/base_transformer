#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import Tensor

from script.layers.positional_encording import PostionalEncodingLayer
from script.layers.layer_embedding import TokenEmbeddingLayer

"""
author by Iwadon
"""


class TransformerEmbeddingBlock(nn.Module):
    """
    埋め込み層本体
    入力されたトークン列
    """

    def __init__(self, vocab_size: int, model_dim: int, input_len: int, drop_prob: float, device: str):
        """
        Transformerにおける埋め込みブロックを初期化します
        :param int vocab_size: 語彙数
        :param int model_dim: モデルの次元数
        :param int input_len: 入力長
        :param float drop_prob: ドロップアウト率
        :param str device: 計算時に利用するデバイス
        """
        super(TransformerEmbeddingBlock, self).__init__()
        # 埋め込みレイヤーの初期化
        self.l_token_embedding = TokenEmbeddingLayer(vocab_size, model_dim)
        # 位置埋め込みレイヤーの初期化
        self.l_positional_encoding = PostionalEncodingLayer(model_dim, input_len, device)
        # ドロップアウト
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: 単語ID列(Tensor) [batch * input_length]
        :return: [batch * input_length * d_model]
        """
        # 埋め込み表現を獲得 out_token_emb: [batch * max_length * d_model]
        out_token_emb = self.l_token_embedding(x)
        # 位置エンコーディングを取得。本処理を行うことでトークン列の位置情報を埋め込みます。out_positional_encode: [input_len * d_model]
        out_positional_encode = self.l_positional_encoding(x)
        # 埋め込み表現に位置エンコーディングを加算し,dropout
        return self.drop_out(out_token_emb + out_positional_encode)
