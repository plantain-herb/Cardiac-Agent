import os
import einops
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
class Color3D_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        loss_dict=dict(type='BceLoss', size_average=False, reduce=False),
    ):
        super(Color3D_Head, self).__init__()
        self.conv_LR = nn.Conv3d(in_channels, 3, 1)  # 分出心脏左半部分，心脏右半部分（分大类 Big Class）
        self.conv_L = nn.Conv3d(in_channels, 2, 1)  # 左半部分细分
        self.conv_R = nn.Conv3d(in_channels, 2, 1) # 右半部分细分
        self.conv_F = nn.Conv3d(in_channels, 2, 1) # 右半部分细分

        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.step = 0
    def forward(self, inputs):
        # print(inputs.shape)
        # raise
        inputs = F.interpolate(inputs, scale_factor=(2.0, ) * 3, mode="trilinear")
        pred_LR = self.conv_LR(inputs)
        pred_L = self.conv_L(inputs)
        pred_R = self.conv_R(inputs)
        pred_F = self.conv_F(inputs)
        return pred_LR, pred_L, pred_R, pred_F

    def neg_soft_mining(self, neg_loss, tp_seg, neg_sample_ratio=[3, 6], min_positive_pt_num=50):
        neg_loss = neg_loss.view(-1)
        neg_point_num = torch.clamp(torch.sum(tp_seg)//150, min=min_positive_pt_num)
        # neg_point_num = torch.sum(tp_seg)//200
        neg_point_num2 = int(torch.clamp(torch.sum(tp_seg)//100, min=min_positive_pt_num))
        loss_select, idxs1 = torch.topk(neg_loss, neg_point_num2, dim=0, largest=True)
        with torch.no_grad():
            perm = torch.randperm(neg_point_num2)
        loss_select = loss_select[perm[: int(neg_point_num)]]

        # import SimpleITK as sitk
        # img2 = sitk.GetImageFromArray(loss_select.cpu().numpy())
        # sitk.WriteImage(img2, '' + '-seg.nii.gz')

        loss_select = loss_select.mean()
        return loss_select

    def loss(self, inputs, targets):
        with torch.no_grad():
            target = targets.long()
            # import time
            # s = str(time.time())
            # import SimpleITK as sitk

            weight_lr = torch.zeros_like(target)
            target_lr = torch.where(target==1, 2, 0) + torch.where(target==2, 2, 0) + torch.where(target==6, 1, 0) + torch.where(target==3, 3, 0) + torch.where(target==4, 3, 0) + torch.where(target==5, 1, 0)
            # sitk.WriteImage(sitk.GetImageFromArray(target_lr[0][0].cpu().float().numpy()), "./"+ s + "lr.nii.gz")
            weight_lr[target_lr == 1] = 5
            weight_lr[target_lr == 2] = 4
            weight_lr[target_lr == 3] = 5
            target_lr_av = (target_lr > 0)
            target_lr = target_lr_av * (target_lr - 1)
            target_lr = F.one_hot(target_lr.long(), 3)
            target_lr = einops.rearrange(target_lr, "b c d h w c1 -> b (c c1) d h w")
            target_lr = target_lr.float()
            weight_lr = weight_lr.float()
            weight_lr_sum = torch.sum(weight_lr) + 1

            weight_l = torch.zeros_like(target)
            target_l = torch.where(target==1, 1, 0) + torch.where(target==2, 2, 0)
            # sitk.WriteImage(sitk.GetImageFromArray(target_l[0][0].cpu().float().numpy()), "./"+ s + "-l.nii.gz")
            weight_l[target_l == 1] = 6
            weight_l[target_l == 2] = 8
            target_l_av = (target_l > 0)
            target_l = target_l_av * (target_l - 1)
            target_l = F.one_hot(target_l.long(), 2)
            target_l = einops.rearrange(target_l, "b c d h w c1 -> b (c c1) d h w")
            target_l = target_l.float()
            weight_l = weight_l.float()
            weight_l_sum = torch.sum(weight_l) + 1

            weight_r = torch.zeros_like(target)
            target_r = torch.where(target==3, 1, 0) + torch.where(target==4, 2, 0)
            # sitk.WriteImage(sitk.GetImageFromArray(target_r[0][0].cpu().float().numpy()), "./"+ s + "-r.nii.gz")
            weight_r[target_r == 1] = 8
            weight_r[target_r == 2] = 20
            target_r_av = (target_r > 0)
            target_r = target_r_av * (target_r - 1)
            target_r = F.one_hot(target_r.long(), 2)
            target_r = einops.rearrange(target_r, "b c d h w c1 -> b (c c1) d h w")
            target_r = target_r.float()
            weight_r = weight_r.float()
            weight_r_sum = torch.sum(weight_r) + 1


            weight_f = torch.zeros_like(target)
            target_f = torch.where(target==5, 1, 0) + torch.where(target==6, 2, 0)
            # sitk.WriteImage(sitk.GetImageFromArray(target_f[0][0].cpu().float().numpy()), "./"+ s + "-f.nii.gz")
            weight_f[target_f == 1] = 8
            weight_f[target_f == 2] = 8
            target_f_av = (target_f > 0)
            target_f = target_f_av * (target_f - 1)
            target_f = F.one_hot(target_f.long(), 2)
            target_f = einops.rearrange(target_f, "b c d h w c1 -> b (c c1) d h w")
            target_f = target_f.float()
            weight_f = weight_f.float()
            weight_f_sum = torch.sum(weight_f) + 1


        loss_lr = focal_loss(inputs[0], target_lr, alpha=0.25, gamma=2.0)
        loss_lr = (loss_lr*weight_lr).sum()/weight_lr_sum
        loss_l = focal_loss(inputs[1], target_l, alpha=0.25, gamma=2.0)
        loss_l = (loss_l*weight_l).sum()/weight_l_sum
        loss_r = focal_loss(inputs[2], target_r, alpha=0.25, gamma=2.0)
        loss_r = (loss_r*weight_r).sum()/weight_r_sum
        loss_f = focal_loss(inputs[3], target_f, alpha=0.25, gamma=2.0)
        loss_f = (loss_f*weight_f).sum()/weight_f_sum
        

        # loss_lr = self.loss_func(inputs[0], target_lr.squeeze(0).long())#分大类
        # loss_lr = (loss_lr*weight_lr).sum()/weight_lr_sum
        # loss_l = self.loss_bce_func(inputs[1], target_l)#分大类
        # loss_l = (loss_l*weight_l).sum()/weight_l_sum
        # loss_r = self.loss_bce_func(inputs[2], target_r)#分大类
        # loss_r = (loss_r*weight_r).sum()/weight_r_sum
        # loss_f = self.loss_bce_func(inputs[3], target_f)#分大类
        # loss_f = (loss_f*weight_f).sum()/weight_f_sum

        return {
                "loss_1": loss_lr*20,
                'loss_2': loss_l*10,
                'loss_3': loss_r*10,
                'loss_4': loss_f*10,
                }

def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

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
        self.loss_lre_func = torch.nn.BCEWithLogitsLoss(reduce=False)
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

        loss_bin = self.loss_lre_func(pred_bin, bin_target)
        loss_bin = loss_bin.mean()

        return {"loss_bin": loss_bin }
