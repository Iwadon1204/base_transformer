#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from blocks.emmbedding_block import TransformerEmbeddingBlock
from blocks.decoder_block import DecoderBlock

from util.model_setting import GlobalModelSetting
from util import const

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        setting = GlobalModelSetting.get_instance()
        # デコーダーのレイヤー数を取得
        dec_layer_num = setting.get_setting(const.KEY_DEC_LAYER_NUM)
        # モデルの次元数を取得
        d_model = setting.get_setting(const.KEY_MODEL_DIM)
        # デコーダーの語彙数を取得
        dec_voc_size = setting.get_setting(const.KEY_DEC_VOCAB_SIZE)

        # 埋め込みブロック
        self.__emb = TransformerEmbeddingBlock()

        # 指定されたレイヤーの数だけブロックを生成する
        self.__layers = nn.ModuleList([DecoderBlock() for _ in range(dec_layer_num)])
        # 最終的な出力となる線形レイヤー
        self.__linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, decoder_input: Tensor, encoder_output: Tensor, decoder_mask: Tensor, encoder_mask: Tensor) -> Tensor:
        """
        :param decoder_input: デコーダーへの入力
        :param encoder_output: エンコーダーからの出力
        :param decoder_mask:　デコーダーへの入力マスク
        :param encoder_mask: エンコーダー出力へのマスク
        :return: 最終的な出力 -> [batch * input_length * vocab_size]
        """
        # Decoder入力ID列を埋め込み表現に変換 x -> [batch_size * input_len * d_model]
        x = self.__emb(decoder_input)

        for layer in self.__layers:
            x = layer(x, encoder_output, decoder_mask, encoder_mask)

        # x -> [batch_size * input_len * d_model]
        # 最終的な出力を線形レイヤーに入力
        output = self.__linear(x)
        return output
