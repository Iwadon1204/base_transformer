#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

from torch import nn, Tensor


class PositionwiseFeedForward(nn.Module):
    """
    順伝播レイヤー
    """

    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        """
        :param d_model: モデルの次元数
        :param hidden: 隠れ層の数
        :param drop_prob: ドロップアウトレート
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x) -> Tensor:
        """
        順伝播
        :param x: [batch * d_model]
        :return: [batch * d_model]
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
