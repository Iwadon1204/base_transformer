"""
@autor Iwadon
"""


import sys
import json
from threading import Lock
from util import const
import constant

class GlobalModelSetting(object):
    __unique_instance = None
    __lock = Lock()
    __setting_data = {}

    def __new__(cls):
        raise NotImplementedError('Cannot initialize Constructor')

    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    @classmethod
    def get_instance(cls):
        """
        GlobalModelSettingクラスのインスタンスを取得する。
        :return:
        """
        if not cls.__unique_instance:
            with cls.__lock:
                if not cls.__unique_instance:
                    cls.__unique_instance = cls.__internal_new__()
                    constant.init_constant()
        return cls.__unique_instance

    def get_setting(self, key):
        """
        指定されたKeyの設定値を取得する。
        :param key: 設定Key
        :return: 設定値(Keyに該当する設定値がない場合、Noneを返却する)
        """
        return self.__setting_data.get(key, None)

    def set_default_setting(self) -> None:
        """
        デフォルトの設定値を設定する。
        """
        self.__setting_data[const.KEY_SRC_PAD_IDX] = 1
        self.__setting_data[const.KEY_TRG_PAD_IDX] = 1
        self.__setting_data[const.KEY_ENC_VOCAB_SIZE] = 32000
        self.__setting_data[const.KEY_DEC_VOCAB_SIZE] = 32000
        self.__setting_data[const.KEY_MODEL_DIM] = 512
        self.__setting_data[const.KEY_HEAD_NUM] = 8
        self.__setting_data[const.KEY_INPUT_LEN] = 512
        self.__setting_data[const.KEY_FFN_HIDDEN_NUM] = 2048
        self.__setting_data[const.KEY_ENC_LAYER_NUM] = 6
        self.__setting_data[const.KEY_DEC_LAYER_NUM] = 6
        self.__setting_data[const.KEY_DROPOUT_RATE] = 0.1
        self.__setting_data[const.KEY_USE_DEVICE] = "cpu"
        self.__setting_data[const.KEY_LAYER_NORM_EPS] = 1e-12

    def load_setting_file(self, file_path) -> None:
        """
        設定ファイルを読み込む。
        :param file_path: ファイルパス
        """
        with open(file_path, encoding="utf-8") as f:
            for line in f.readlines():
                self.__setting_data = json.loads(line)