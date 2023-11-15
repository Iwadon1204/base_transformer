#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

from script.blocks.emmbedding_block import TransformerEmbeddingBlock
from script.blocks.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
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
        self.emb = TransformerEmbeddingBlock(vocab_size=dec_voc_size, model_dim=d_model, input_length=max_len,
                                             drop_prob=drop_prob, device=device)

        # 指定されたレイヤーの数だけブロックを生成する
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, decoder_input, encoder_output, decoder_mask, encoder_mask) -> Tensor:
        """
        :param decoder_input: デコーダーへの入力
        :param encoder_output: エンコーダーからの出力
        :param decoder_mask:　デコーダーへの入力マスク
        :param encoder_mask: エンコーダー出力へのマスク
        :return: 最終的な出力 -> [batch * max_length * vocab_size]
        """
        # 入力ID列を埋め込み表現に変換
        x = self.emb(decoder_input)
        # x -> [batch_size * input_len * d_model]

        for layer in self.layers:
            x = layer(x, encoder_output, decoder_mask, encoder_mask)

        # x -> [batch_size * input_len * d_model]
        # 最終的な出力を線形レイヤーに入力
        output = self.linear(x)

        return output
