#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor

"""
author by Iwasaki
"""

class PostionalEncodingLayer(nn.Module):
    """
    位置エンコーディングのためのレイヤー
    """

    def __init__(self, model_dim: int, input_max_length: int, device: str):
        """
        :param model_dim: 隠れ層の次元数
        :param input_max_length: 最大入力長
        :param device: 利用するデバイス
        """
        super(PostionalEncodingLayer, self).__init__()

        # 入力シーケンス数 * 次元数の行列を初期化
        self.encoding = torch.zeros(input_max_length, model_dim, device=device)

        # 勾配を計算しない
        self.encoding.requires_grad = False

        # 位置情報の等差数列作成[0, 1, 2,...,input_length-1]
        pos = torch.arange(0, input_max_length, device=device)

        # 行列の形式を変換　input_lengthが256の場合　[256] -> [256, 1]
        pos = pos.float().unsqueeze(dim=1)

        # 次元数までの偶数の等差数列を作成
        _2i = torch.arange(0, model_dim, step=2, device=device).float()

        # 全ての行の偶数列目にsin(pos / (10000^(2i/model_dim)))を代入
        # 全ての行の奇数列目にcos(pos / (10000^(2i/model_dim)))を代入
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / model_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / model_dim)))


    def forward(self, x: Tensor) -> Tensor:
        """
        位置エンコーディングを獲得する
        :param x: 入力[batch * max_length]
        :return: positional encoding行列[max_length * d_model]
        """

        # バッチサイズ, 実際に入力された入力長
        _, seq_len = x.size()
        # 入力長に応じたのpositional encoding行列を返却
        return self.encoding[:seq_len, :]
