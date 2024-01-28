#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author Iwadon

本プログラムで利用する定数のリスト
"""
from util import const

def init_constant():
    try:
        if const.IS_INIT:
            # 既に初期化済
            return
    except:
        # 設定未完了
        pass

    ###########  設定完了フラグ  ##############
    const.IS_INIT = True

    ###########  設定Key  ##############
    const.KEY_SRC_PAD_IDX = "src_pad_idx"
    const.KEY_TRG_PAD_IDX = "trg_pad_idx"
    const.KEY_ENC_VOCAB_SIZE = "enc_voc_size"
    const.KEY_DEC_VOCAB_SIZE = "dec_voc_size"
    const.KEY_MODEL_DIM = "d_model"
    const.KEY_HEAD_NUM = "head_num"
    const.KEY_INPUT_LEN = "input_len"
    const.KEY_FFN_HIDDEN_NUM = "ffn_hidden_num"
    const.KEY_ENC_LAYER_NUM = "encoder_layer_num"
    const.KEY_DEC_LAYER_NUM = "decoder_layer_num"
    const.KEY_DROPOUT_RATE = "dropout_rate"
    const.KEY_USE_DEVICE = "device"
    const.KEY_LAYER_NORM_EPS = "eps"

