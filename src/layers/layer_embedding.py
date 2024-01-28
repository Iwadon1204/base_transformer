#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from util.model_setting import GlobalModelSetting
from util import const

"""
@autor Iwadon
"""

class TokenEmbeddingLayer(nn.Embedding):
    """
    pytorchのnn.Embeddingを利用して、トークンの埋め込み表現を作成します。
    """

    def __init__(self):
        super_instance = super(TokenEmbeddingLayer, self)
        setting = GlobalModelSetting.get_instance()
        # モデルの次元数を取得
        d_model = setting.get_setting(const.KEY_MODEL_DIM)
        # 語彙サイズ
        vocab_size = setting.get_setting(const.KEY_ENC_VOCAB_SIZE)
        # 特殊トークンのIDを設定
        src_pad_idx = setting.get_setting(const.KEY_SRC_PAD_IDX)
        super_instance.__init__(vocab_size, d_model, padding_idx=src_pad_idx)