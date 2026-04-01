import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from custom.model.utils import build_loss
from custom.model.registry import HEADS, LOSSES


class FocalLoss_Sigmoid(nn.Module):
    """
    Example:
    .. code-block:: python

        >>> import torch
        >>> Loss = FocalLoss_Sigmoid(alpha=0.5, gamma=2.0)
        >>> inputs = torch.rand(1, 2, 3)
        >>> targets = torch.ones(1, 1, 3).long()
        >>> loss = Loss(inputs, targets)
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2, eps: float = 1e-12) -> None:
        """
        FocalLoss_Sigmoid, sigmoid方式的focal loss, 将每个类别看成二分类计算损失函数，使用sigmoid计算类别概率
        Args:
            alpha:　float,
            gamma: float,
            eps:　float, 防止分母为0
        """
        super(FocalLoss_Sigmoid, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): [N, C, ...], 未进行sigmoid的inputs
            targets (torch.Tensor): [N, 1, ...],
        """

        p = torch.sigmoid(inputs)
        num_classes = p.shape[1]
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(0, num_classes, dtype=dtype,
                                   device=device).repeat(targets.transpose(1, -1).shape).transpose(1, -1)
        term1 = (1 - p)**self.gamma * torch.log(p + self.eps)
        term2 = p**self.gamma * torch.log(1 - p + self.eps)
        loss = -(targets == class_range).float() * term1 * self.alpha - ((targets != class_range) * (targets >= 0)
                                                                         ).float() * term2 * (1 - self.alpha)
        return loss


@HEADS.register_module()
class Seg_Head_Heart(nn.Module):
    def __init__(
        self, in_channels: int, scale_factor,
    ):
        super(Seg_Head_Heart, self).__init__()
        # TODO: 定制Head模型
        self.conv_bin = nn.Conv3d(in_channels, 1, 1)
        # self.conv_tumor = nn.Conv3d(in_channels, 1, 1)
        # self.conv_abdomen = nn.Conv3d(in_channels, 5, 1)
        self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        #self.loss_ce_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.scale_factor = scale_factor
        self._show_count = 0

    def forward(self, inputs):
        # TODO: 定制forward网络
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="trilinear", align_corners=True)
        pred_bin = self.conv_bin(inputs)
        # pred_tumor = self.conv_tumor(inputs)
        # pred_abdomen = self.conv_abdomen(inputs)
        return pred_bin

    def loss(self, inputs, targets):
        pred_bin = inputs
        seg  = targets
        with torch.no_grad():
            # data_type = data_type[:, :, None, None, None]

            bin_target = seg == 1
            # bin_target |= (seg == 2) & (data_type == 0)

            # tumor_target = seg == 2
            # tumor_av = bin_target & (data_type == 0)
            # tumor_av = tumor_av * 1.0
            # tumor_av_count = tumor_av.sum()
            # if tumor_av_count == 0:
            #     tumor_av_count = 1

            # abdomen_av = (data_type == 1) * 1.0
            # abdomen_av_count = abdomen_av.sum()
            # if abdomen_av_count == 0:
            #     abdomen_av_count = 1

            bin_target = bin_target * 1.0
            # tumor_target = tumor_target * 1.0
            # abdomen_target = seg.long()

        loss_bin = self.loss_bce_func(pred_bin, bin_target)
        loss_bin = loss_bin.mean()

        return {"loss_bin": loss_bin }
