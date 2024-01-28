#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

from torch import nn, Tensor
from util.model_setting import GlobalModelSetting
from util import const


class PositionwiseFeedForward(nn.Module):
    """
    順伝播レイヤー
    """

    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        setting = GlobalModelSetting.get_instance()

        # モデルの次元数を取得
        d_model = setting.get_setting(const.KEY_MODEL_DIM)
        # 隠れ層の数を取得
        ffn_hidden = setting.get_setting(const.KEY_FFN_HIDDEN_NUM)
        # ドロップアウト率を取得
        drop_prob = setting.get_setting(const.KEY_DROPOUT_RATE)

        self.__linear1 = nn.Linear(d_model, ffn_hidden)
        self.__linear2 = nn.Linear(ffn_hidden, d_model)
        self.__relu = nn.ReLU()
        self.__dropout = nn.Dropout(p=drop_prob)

    def forward(self, x) -> Tensor:
        """
        :param x: [batch * d_model]
        :return: [batch * d_model]
        """
        x = self.__linear1(x)
        x = self.__relu(x)
        x = self.__dropout(x)
        x = self.__linear2(x)
        return x
