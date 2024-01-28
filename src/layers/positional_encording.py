#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor

from util.model_setting import GlobalModelSetting
from util import const

"""
@autor Iwadon
"""

class PostionalEncodingLayer(nn.Module):
    """
    位置エンコーディングのためのレイヤー
    """

    def __init__(self):
        super(PostionalEncodingLayer, self).__init__()

        setting = GlobalModelSetting.get_instance()

        # モデルの次元数を取得
        d_model = setting.get_setting(const.KEY_MODEL_DIM)

        # モデルの入力長を取得
        input_len = setting.get_setting(const.KEY_INPUT_LEN)

        # 計算時利用するデバイス
        device = setting.get_setting(const.KEY_USE_DEVICE)

        # 入力シーケンス数 * 次元数の行列を初期化
        self.__encoding = torch.zeros(input_len, d_model, device=device)

        # 勾配を計算しない
        self.__encoding.requires_grad = False

        # 位置情報の等差数列作成[0, 1, 2,...,input_length-1]
        pos = torch.arange(0, input_len, device=device)

        # 行列の形式を変換　input_lengthが256の場合　[256] -> [256, 1]
        pos = pos.float().unsqueeze(dim=1)

        # 次元数までの偶数の等差数列を作成
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 全ての行の偶数列目にsin(pos / (10000^(2i/model_dim)))を代入
        # 全ての行の奇数列目にcos(pos / (10000^(2i/model_dim)))を代入
        self.__encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.__encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))


    def forward(self, x: Tensor) -> Tensor:
        """
        位置エンコーディングを獲得する
        :param x: 入力[batch * max_length]
        :return: positional encoding行列[max_length * d_model]
        """

        # バッチサイズ, 実際に入力された入力長
        _, seq_len = x.size()
        # 入力長に応じたのpositional encoding行列を返却
        return self.__encoding[:seq_len, :]
