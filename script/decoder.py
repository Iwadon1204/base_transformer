#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from blocks.emmbedding_block import TransformerEmbeddingBlock
from blocks.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, dec_voc_size: int, input_len: int, d_model: int, ffn_hidden: int, n_head: int, n_layers: int, drop_prob: int, device: str):
        """
        :param dec_voc_size: 語彙サイズ
        :param max_len: 入力長
        :param d_model: 次元数
        :param ffn_hidden: FFNレイヤーの隠れ層
        :param n_head: ヘッドの数
        :param n_layers: レイヤー数
        :param drop_prob: ドロップアウト率
        :param device: 計算に利用するデバイス
        """
        super(Decoder, self).__init__()
        # 埋め込みブロック
        self.emb = TransformerEmbeddingBlock(vocab_size=dec_voc_size, model_dim=d_model, input_length=input_len,
                                             drop_prob=drop_prob, device=device)

        # 指定されたレイヤーの数だけブロックを生成する
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        # 最終的な出力となる線形レイヤー
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, decoder_input: Tensor, encoder_output: Tensor, decoder_mask: Tensor, encoder_mask: Tensor) -> Tensor:
        """
        :param decoder_input: デコーダーへの入力
        :param encoder_output: エンコーダーからの出力
        :param decoder_mask:　デコーダーへの入力マスク
        :param encoder_mask: エンコーダー出力へのマスク
        :return: 最終的な出力 -> [batch * max_length * vocab_size]
        """
        # Decoder入力ID列を埋め込み表現に変換 x -> [batch_size * input_len * d_model]
        x = self.emb(decoder_input)

        for layer in self.layers:
            x = layer(x, encoder_output, decoder_mask, encoder_mask)

        # x -> [batch_size * input_len * d_model]
        # 最終的な出力を線形レイヤーに入力
        output = self.linear(x)
        return output
