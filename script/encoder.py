#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch.nn as nn
from torch import Tensor

import sys
sys.path.append('..')
aaaa = sys.path
from script.blocks.emmbedding_block import TransformerEmbeddingBlock
from script.blocks.encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    エンコーダーモデル
    """
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        :param enc_voc_size: 語彙サイズ
        :param max_len: 入力長
        :param d_model: 次元数
        :param ffn_hidden: FFNの隠れ層
        :param n_head: ヘッド数
        :param n_layers: レイヤー数
        :param drop_prob: ドロップアウト率
        :param device: 学習時利用デバイス
        """
        super(Encoder, self).__init__()
        # 埋め込みブロック
        self.emb = TransformerEmbeddingBlock(vocab_size=enc_voc_size, model_dim=d_model, input_length=max_len, drop_prob=drop_prob, device=device)

        # Encoderブロックをレイヤーの数だけ作成
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x: Tensor, encoder_mask: Tensor) -> Tensor:
        """
        :param x: 入力トークン列(Tensor) [batch * max_length]
        :param encoder_mask: マスク(Tensor) [batch * 1 * 1 * max_length]
        :return: Encoderモデル出力(Tensor) [batch * max_length * d_model]
        """
        # 埋め込み層によって獲得されるTensor[batch * max_length * d_model]
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, encoder_mask)
        return x