#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import time
import random

import torch
import sentencepiece as sp
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn

############### LOCAL ENV ####################
MY_TRANSFORMER_PATH = r"../src"
SENTENCE_PIECE_DATA_PATH = r'Please input your env'
TRAIN_DATA_PATH = r"../data/XXXXX"
SAVE_PATH = "model.ml"

sys.path.append(MY_TRANSFORMER_PATH)
from transformer import Transformer
from util.model_setting import GlobalModelSetting
from util import const

tokenizer = sp.SentencePieceProcessor()
tokenizer.load(SENTENCE_PIECE_DATA_PATH)

PAD_ID = tokenizer.PieceToId("[PAD]")
CLS_ID = tokenizer.PieceToId("[CLS]")

VOCAB_SIZE = tokenizer.vocab_size()

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCH_NUM = 10

setting = GlobalModelSetting.get_instance()
setting.set_default_setting()

INPUT_LENGTH = setting.get_setting(const.KEY_INPUT_LEN)

VALID_DATASET_RATE = 0.999

use_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SampleDataSet(Dataset):
    """
    サンプル用のデータセットクラス
    """
    __data = []

    def __init__(self, basedata):
        self.__data = basedata

    def __getitem__(self, idx):
        input_text = self.__data[idx]["input"].lower()  # 入力文
        target_text = self.__data[idx]["target"].lower()  # 出力文

        encoded_enc_input = tokenizer.Encode(input_text, add_bos=True, add_eos=True)
        encoded_dec_input = tokenizer.Encode(target_text, add_bos=True, add_eos=True)
        encoded_target = tokenizer.Encode(target_text, add_bos=False, add_eos=True)

        encoded_enc_input.extend([PAD_ID] * (INPUT_LENGTH - len(encoded_enc_input)))
        encoded_dec_input.extend([PAD_ID] * (INPUT_LENGTH - len(encoded_dec_input)))
        encoded_target.extend([PAD_ID] * (INPUT_LENGTH - len(encoded_target)))

        enc_input_ids = torch.tensor(encoded_enc_input, dtype=torch.int32)
        dec_input_ids = torch.tensor(encoded_dec_input, dtype=torch.int32)
        tgt_ids = torch.tensor(encoded_target, dtype=torch.int32)

        return enc_input_ids, dec_input_ids, tgt_ids

    def __len__(self):
        return len(self.__data)

def exec_train(dataloader, model, loss_fn, optimizer):
    train_loss = 0
    size = len(dataloader.dataset)
    for enc_input_ids, dec_input_ids, tgt_ids in dataloader:
        mem_enc_input_ids = enc_input_ids.to(use_device)
        mem_dec_input_ids = dec_input_ids.to(use_device)
        mem_tgt_ids = tgt_ids.to(use_device)

        preds = model(mem_enc_input_ids, mem_dec_input_ids)  # 予測計算
        pred_trans = preds.view(-1, VOCAB_SIZE).to("cpu")
        target_trans = mem_tgt_ids.view(-1).long().to("cpu")
        loss = loss_fn(pred_trans, target_trans)  # 誤差計算
        optimizer.zero_grad()
        loss.backward()  # 誤差伝播
        optimizer.step()  # パラメータ更新
        train_loss += loss.item()

    del mem_enc_input_ids, mem_dec_input_ids, mem_tgt_ids
    print("loss avg: " + str(train_loss / size))


def exec_validation(dataloader, model, loss_fn):
    pass


if __name__ == '__main__':

    ################ データ準備 ################
    base_data = []
    with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
        for line in f.readlines():
            row_data = json.loads(line)
            base_data.append(row_data)

    dataset = SampleDataSet(base_data)

    model = Transformer()
    print("base_dataset length : " + str(len(dataset)))
    # 分割するインデックス
    split_index = round(len(dataset) * (1-VALID_DATASET_RATE))
    # 学習用と検証用のデータセットに分割する
    train_data, valid_data = torch.utils.data.random_split(
        dataset,
        [split_index, len(dataset) - split_index]
    )
    print("train dataset length : " + str(len(train_data)))
    try:
        # 学習用データローダー
        train_dataloader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2,
                                      pin_memory=True)
        # 検証用データローダー
        valid_dataloader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=2,
                                      pin_memory=True)
        model.to(use_device)

        # 最適化関数
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 損失関数
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        for t in range(EPOCH_NUM):
            print("----------------------- epoch : " + str(t) + " ------------------------")
            time_sta = time.time()
            exec_train(train_dataloader, model, loss_fn, optimizer)
            exec_validation(valid_dataloader, model, loss_fn)
            time_end = time.time()
            # 経過時間（秒）
            time_log = time_end - time_sta
            print("elapsed time : " + str(time_log))
            torch.save(model.state_dict(), SAVE_PATH)

    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
    finally:
        del model
        if use_device == 'cuda':
            torch.cuda.empty_cache()
    print("Sample Finish.")
