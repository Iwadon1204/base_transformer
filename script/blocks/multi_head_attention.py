#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from script.layers.scale_dot_product_attention_layer import ScaleDotProductAttention

class MultiHeadAttentionBlock(nn.Module):
    """
    MaltiHeadAttentionを実現するためのブロック
    """
    def __init__(self, d_model: int, n_head: int, dropout_rate: float):
        """
        コンストラクタ
        :param int d_model: モデルの次元数
        :param int n_head: マルチヘッド数
        :param float dropout_rate:
        """
        super(MultiHeadAttentionBlock, self).__init__()
        # Head数
        self.head_num = n_head
        self.attention = ScaleDotProductAttention()

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        # 入力されたQuery, Key, Valueに全結合レイヤーを通し、指定されたHead数に分割する
        query = self.split(self.linear_q(q))
        key = self.split(self.linear_k(k))
        value = self.split(self.linear_v(v))

        # 各HeadのAttentionを計算
        att_val, att_weight = self.attention(query, key, value, mask)

        # 分割したものを結合
        out = self.concat(att_val)

        # 全結合レイヤーを通し最終的な本ブロックの出力とする
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        """
        パラメータに応じて、入力されたTensorを分割する
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        # headの数だけ分割
        d_tensor = d_model // self.head_num

        # 変形
        tensor = tensor.view(batch_size, length, self.head_num, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
        splitによって分割されたものを結合する
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
