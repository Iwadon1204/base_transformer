#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

"""
@autor Iwadon
"""

class TokenEmbeddingLayer(nn.Embedding):
    """
    pytorchのnn.Embeddingを利用して、トークンの埋め込み表現を作成します。
    """

    def __init__(self, vocab_size: int, model_dim: int, pad_idx: int = 0) -> None:
        """
        :param vocab_size: 語彙数
        :param d_model: モデルの次元数
        """
        super(TokenEmbeddingLayer, self).__init__(vocab_size, model_dim, padding_idx=pad_idx)