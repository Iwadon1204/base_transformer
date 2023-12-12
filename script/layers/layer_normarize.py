import torch
from torch import nn, Tensor

"""
@autor Iwadon
"""


class LayerNorm(nn.Module):
    """
    Normalizeレイヤー
    """

    def __init__(self, d_model: int, eps: float = 1e-12):
        """
        :param d_model: モデルの次元数
        :param eps: eps
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

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
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
