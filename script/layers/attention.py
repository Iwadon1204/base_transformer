#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

"""
author by Iwadon
"""

class Attention(nn.Module):
    """
    Attentionクラス
    """

    def __init__(self, query_length: int, depth: int):
        """

        :param query_length: クエリ長
        :param depth: クエリ次元数
        """
        super(Attention, self).__init__()
        self.depth = depth
        self.query_length = query_length
        self.query_linear_layer = nn.Linear(query_length, depth)