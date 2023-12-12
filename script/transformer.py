#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch
from torch import nn, Tensor

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """
    Transformerモデル
    """

    def __init__(self, src_pad_idx: int, trg_pad_idx: int, enc_voc_size: int, dec_voc_size: int,
                 d_model: int, n_head: int, input_len: int,
                 ffn_hidden: int, n_layers, drop_prob: float, device: str):
        """
        :param src_pad_idx: 入力におけるPADトークンのID
        :param trg_pad_idx: 出力におけるPADトークンのID
        :param enc_voc_size: エンコーダーの語彙サイズ
        :param dec_voc_size: デコーダーの語彙サイズ
        :param d_model: モデルの次元数
        :param n_head: ヘッドの数
        :param input_len: 入力長
        :param ffn_hidden: FFNレイヤーの層数
        :param n_layers: Encoder/Decoderのレイヤー数
        :param drop_prob: ドロップアウト率
        :param device: 学習時に利用するデバイス
        """
        super().__init__()
        # 特殊トークンのIDを設定
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        # 計算時利用するデバイス
        self.device = device

        # エンコーダーモデル
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               input_len=input_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # デコーダーモデル
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               input_len=input_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        """
        :param src: 入力トークン列(Tensor) [batch * max_length]
        :param trg: ターゲットトークン列(Tensor) [batch * max_length]
        :return:　transformerモデルの出力(Tensor)[batch * max_length * vocab_size]
        """
        # Encoder入力ID列からマスクを作成
        src_mask = self.make_enc_mask(src)
        # DecoderID列からマスクを作成
        trg_mask = self.make_dec_mask(trg)
        # エンコーダーモデルに入力
        enc_out = self.encoder(src, src_mask)
        # エンコーダーからの出力、ターゲットをデコーダーモデルに入力
        output = self.decoder(trg, enc_out, trg_mask, src_mask)
        return output

    def make_enc_mask(self, src: Tensor) -> Tensor:
        """
        Encoder self-attentionで利用するMASKを作成する
        入力トークン列のPAD部分をMASK
        :param src: 入力トークン列(Tensor) [batch * max_length]
        :return: マスク(Tensor) [batch * 1 * 1 * max_length]
        """
        # PADの箇所をMASKする
        enc_mask = (src == self.src_pad_idx)
        # 以降の処理のためにTensorの形式を変換
        enc_mask = enc_mask.unsqueeze(1).unsqueeze(2)
        return enc_mask

    def make_dec_mask(self, trg: Tensor) -> Tensor:
        """
        Decoderで利用するMASKを作成する
        マスクは【PAD】および【未来の系列】に関して参照できないようにマスクを作成する
        :param trg: ターゲットトークン列(Tensor) [batch * max_length]
        :return: [batch * 1 * max_length * max_length]
        """
        # PAD部分のマスク trg_pad_mask -> [batch * 1 * max_length * 1]
        dec_pad_mask = (trg == self.trg_pad_idx).unsqueeze(1)

        trg_len = trg.shape[1]
        # 入力長に応じた単位行列を作成
        identity_mat = torch.ones(trg_len, trg_len)
        # 下三角の要素が1の行列に
        upper_triangular_mat = torch.tril(identity_mat)
        # ByteTensorに変換
        dec_sub_mask = upper_triangular_mat.type(torch.ByteTensor)
        # True/FalseのTensorに変換
        dec_sub_mask = (dec_sub_mask == 0).to(self.device)
        # PADマスクと未知マスクのORをとる
        dec_mask = dec_pad_mask | dec_sub_mask
        # MultiHead用に形式変換
        dec_mask = dec_mask.unsqueeze(1)
        return dec_mask

    def get_encoder(self) -> Encoder:
        """
        Encoderモデルを取得する
        :return: エンコーダーモデル
        """
        return self.encoder

    def get_decoder(self) -> Decoder:
        """
        Decoderモデルを取得する
        :return: デコーダーモデル
        """
        return self.decoder
