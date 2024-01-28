#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@autor Iwadon
"""

import math
from torch import nn, Tensor


class ScaleDotProductAttention(nn.Module):
    """
    ScaleDotProductAttention(縮小付内積Attention)レイヤー
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.__softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        """
        :param q: Query[バッチサイズ, Query長, 次元数]
        :param k: key[バッチサイズ, ヘッド数, key長, 次元数]
        :param v: Value[バッチサイズ, value長, 次元数]
        :param mask: attention mask マスクし、attention weightを無視させる部分を0に指定
        :return:
        """
        # [バッチサイズ, Query長, 次元数]
        batch_size, head, length, d_tensor = k.size()

        # 転置
        k_t = k.transpose(2, 3)  # transpose

        # queryとkeyの行列積を√次元数でスケール
        # もし、スケールしなかった場合、行列積の値が大きくなり、Softmaxを通した時に勾配が限りなく0に近づいてしまう
        logit = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # mask処理
        if mask is not None:
            # maskすべき部分は-10000に
            # この処理を行うことで、attention_wightにて無視すべき部分を-∞に
            logit = logit.masked_fill(mask, float('-10000'))
        # softmax
        # ここで出力されるattention_weightが、センテンスにおける関連度/注目度である
        attention_weight = self.__softmax(logit)

        # attention_weightとvalueの行列積
        output_attention = attention_weight @ v

        return output_attention, attention_weight
