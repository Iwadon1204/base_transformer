#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import torch
from torch import nn, Tensor

from encoder import Encoder
from decoder import Decoder

from util.model_setting import GlobalModelSetting
from util import const


class Transformer(nn.Module):
    """
    Transformerモデル
    """

    def __init__(self):
        super().__init__()
        setting = GlobalModelSetting.get_instance()
        # 特殊トークンのIDを設定
        self.__src_pad_idx = setting.get_setting(const.KEY_SRC_PAD_IDX)
        self.__trg_pad_idx = setting.get_setting(const.KEY_TRG_PAD_IDX)
        # 計算時利用するデバイス
        self.__device = setting.get_setting(const.KEY_USE_DEVICE)

        # エンコーダーモデル
        self.__encoder = Encoder()

        # デコーダーモデル
        self.__decoder = Decoder()

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
        enc_out = self.__encoder(src, src_mask)
        # エンコーダーからの出力、ターゲットをデコーダーモデルに入力
        output = self.__decoder(trg, enc_out, trg_mask, src_mask)
        return output

    def make_enc_mask(self, src: Tensor) -> Tensor:
        """
        Encoder self-attentionで利用するMASKを作成する
        入力トークン列のPAD部分をMASK
        :param src: 入力トークン列(Tensor) [batch * max_length]
        :return: マスク(Tensor) [batch * 1 * 1 * max_length]
        """
        # PADの箇所をMASKする
        enc_mask = (src == self.__src_pad_idx)
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
        dec_pad_mask = (trg == self.__trg_pad_idx).unsqueeze(1)

        trg_len = trg.shape[1]
        # 入力長に応じた単位行列を作成
        identity_mat = torch.ones(trg_len, trg_len)
        # 下三角の要素が1の行列に
        upper_triangular_mat = torch.tril(identity_mat)
        # ByteTensorに変換
        dec_sub_mask = upper_triangular_mat.type(torch.ByteTensor)
        # True/FalseのTensorに変換
        dec_sub_mask = (dec_sub_mask == 0).to(self.__device)
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
        return self.__encoder

    def get_decoder(self) -> Decoder:
        """
        Decoderモデルを取得する
        :return: デコーダーモデル
        """
        return self.__decoder
