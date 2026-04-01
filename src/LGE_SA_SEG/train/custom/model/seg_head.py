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
        self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.scale_factor = scale_factor
        self._show_count = 0

    def forward(self, inputs):
        # TODO: 定制forward网络
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="trilinear", align_corners=True)
        pred_bin = self.conv_bin(inputs)
        return pred_bin

    def loss(self, inputs, targets):
        pred_bin = inputs
        seg  = targets
        with torch.no_grad():
            bin_target = seg == 1
            bin_target = bin_target * 1.0

        loss_bin = self.loss_bce_func(pred_bin, bin_target)
        loss_bin = loss_bin.mean()

        return {"loss_bin": loss_bin }


@HEADS.register_module()
class Seg_Head_Heart2d(nn.Module):
    def __init__(
        self, in_channels: int, scale_factor,
    ):
        super(Seg_Head_Heart2d, self).__init__()
        # TODO: 定制Head模型
        self.conv_bin = nn.Conv2d(in_channels, 1, 1)
        self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.scale_factor = scale_factor
        self._show_count = 0

    def forward(self, inputs):
        # TODO: 定制forward网络
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        pred_bin = self.conv_bin(inputs)
        return pred_bin

    def loss(self, inputs, targets):
        pred_bin = inputs
        seg  = targets
        with torch.no_grad():
            bin_target = seg == 1
            bin_target = bin_target * 1.0
            weight = torch.ones_like(bin_target)
            weight = weight + (((bin_target > 0).float()) * 2)

        loss_bin = self.loss_bce_func(pred_bin, bin_target)
        loss_bin = loss_bin * weight
        loss_bin = torch.sum(loss_bin) / (torch.sum(weight) + 1)

        return {"loss_bin": loss_bin }

@HEADS.register_module()
class Seg_Head_Heart2d_multi(nn.Module):
    def __init__(
        self, in_channels: int, scale_factor,
    ):
        super(Seg_Head_Heart2d_multi, self).__init__()
        # TODO: 定制Head模型
        self.conv_bin_1 = nn.Conv2d(in_channels, 1, 1)
        self.conv_bin_2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_bin_3 = nn.Conv2d(in_channels, 1, 1)
        self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.scale_factor = scale_factor
        self._show_count = 0

    def forward(self, inputs):
        # TODO: 定制forward网络
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        pred_bin_1 = self.conv_bin_1(inputs)
        pred_bin_2 = self.conv_bin_2(inputs)
        pred_bin_3 = self.conv_bin_3(inputs)
        return pred_bin_1, pred_bin_2, pred_bin_3

    def loss(self, inputs, targets):
        pred_bin_1,  pred_bin_2, pred_bin_3 = inputs[0], inputs[1], inputs[2]
        seg  = targets
        print(torch.unique(seg))
        with torch.no_grad():
            mask_1 = (seg == 1).float() 
            mask_2 = (seg > 1).float() 
            mask_3 = (seg == 3).float() 
            
            weight_1 = torch.ones_like(mask_1) + (mask_1 * 2)   # 标签1的权重设置
            weight_2 = torch.ones_like(mask_2) + (mask_2 * 2)  # 标签2的权重设置
            weight_3 = torch.ones_like(mask_3) + (mask_3 * 5)  # 标签3的权重设置

        loss_bin_1 = self.loss_bce_func(pred_bin_1, mask_1)
        loss_bin_1 = loss_bin_1 * weight_1
        loss_bin_1 = torch.sum(loss_bin_1) / (torch.sum(weight_1) + 1)

        loss_bin_2 = self.loss_bce_func(pred_bin_2, mask_2)
        loss_bin_2 = loss_bin_2 * weight_2
        loss_bin_2 = torch.sum(loss_bin_2) / (torch.sum(weight_2) + 1)

        loss_bin_3 = self.loss_bce_func(pred_bin_3, mask_3)
        loss_bin_3 = loss_bin_3 * weight_3
        loss_bin_3 = torch.sum(loss_bin_3) / (torch.sum(weight_3) + 1)

        return {"loss_bin_1": loss_bin_1, "loss_bin_2": loss_bin_2, "loss_bin_3": loss_bin_3}