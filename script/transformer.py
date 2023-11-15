#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@autor Iwadon
"""

import sys
import torch
from torch import nn, Tensor

sys.path.append('..')
from script.encoder import Encoder
from script.decoder import Decoder


class Transformer(nn.Module):
    """
    Transformerモデル
    """

    def __init__(self, src_pad_idx: int, trg_pad_idx: int, trg_sos_idx: int, enc_voc_size: int, dec_voc_size: int,
                 d_model: int, n_head: int, input_len: int,
                 ffn_hidden: int, n_layers, drop_prob: float, device: str):
        """

        :param src_pad_idx: 入力におけるPADトークンのID
        :param trg_pad_idx: 出力におけるPADトークンのID
        :param trg_sos_idx: 出力のBOSトークンのID
        :param enc_voc_size: エンコーダーの語彙サイズ
        :param dec_voc_size: デコーダーの語彙サイズ
        :param d_model: モデルの次元数
        :param n_head: ヘッドの数
        :param input_len: 時系列長
        :param ffn_hidden: FFNレイヤーの層数
        :param n_layers: Encoder/Decoderのレイヤー数
        :param drop_prob: ドロップアウト率
        :param device: 学習時に利用するデバイス
        """
        super().__init__()
        # 特殊トークンのIDを設定
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        # 計算時利用するデバイス
        self.device = device

        # エンコーダーモデル
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=input_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # デコーダーモデル
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=input_len,
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
        # 入力ID列からマスクを作成
        src_mask = self.make_src_mask(src)
        # ターゲットID列からマスクを作成
        trg_mask = self.make_trg_mask(trg)
        # エンコーダーモデルに入力
        enc_src = self.encoder(src, src_mask)
        # エンコーダーからの出力、ターゲットをデコーダーモデルに入力
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src: Tensor) -> Tensor:
        """
        入力トークン列のPAD部分のマスクを作成する
        :param src: 入力トークン列(Tensor) [batch * max_length]
        :return: マスク(Tensor) [batch * 1 * 1 * max_length]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg: Tensor) -> Tensor:
        """
        ターゲットID列のマスクを作成する。
        マスクは【PAD】および【未来の系列】に関して参照できないようにマスクを作成する
        :param trg: ターゲットトークン列(Tensor) [batch * max_length]
        :return: [batch * 1 * max_length * max_length]
        """
        # trg_pad_mask -> [batch * 1 * max_length * 1]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        # 入力長に応じた単位行列を作成
        identity_mat = torch.ones(trg_len, trg_len)
        # 上三角の要素が1の行列に
        upper_triangular_mat = torch.tril(identity_mat)
        # ByteTensorに変換
        trg_sub_mask = upper_triangular_mat.type(torch.ByteTensor)
        # True/FalseのTensorに変換
        trg_sub_mask = (trg_sub_mask == 0).to(self.device)
        # PADマスクと未知マスクのAndをとる
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def get_encoder(self) -> Encoder:
        """
        Encoderモデルを取得する
        :return: エンコーダーモデル
        """
        return self.encoder

    def get_encoder(self) -> Decoder:
        """
        Decoderモデルを取得する
        :return: デコーダーモデル
        """
        return self.decoder
