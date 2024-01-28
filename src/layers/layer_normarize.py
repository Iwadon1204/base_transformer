import torch
from torch import nn, Tensor

from util.model_setting import GlobalModelSetting
from util import const

"""
@autor Iwadon
"""


class LayerNorm(nn.Module):
    """
    Normalizeレイヤー
    """

    def __init__(self):
        super(LayerNorm, self).__init__()
        setting = GlobalModelSetting.get_instance()
        # モデルの次元数を取得
        d_model = setting.get_setting(const.KEY_MODEL_DIM)

        # モデルの次元数を取得
        eps = setting.get_setting(const.KEY_LAYER_NORM_EPS)

        self.__gamma = nn.Parameter(torch.ones(d_model))
        self.__beta = nn.Parameter(torch.zeros(d_model))
        self.__eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [batch * d_model]
        :return: [batch * d_model]
        """
        # 平均を求める
        mean = x.mean(-1, keepdim=True)
        # 不編分散(標本分散)を求める
        var = x.var(-1, unbiased=False, keepdim=True)

        # 平均と標準偏差で正規化
        out = (x - mean) / torch.sqrt(var + self.__eps)
        out = self.__gamma * out + self.__beta
        return out
