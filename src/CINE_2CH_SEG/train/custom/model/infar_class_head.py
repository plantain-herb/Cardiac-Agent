import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from custom.model.utils import build_loss
from custom.model.registry import HEADS, LOSSES

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def _linear_combination(self, x, y, epsilon):
        return epsilon*x + (1-epsilon)*y

    def _reduce_loss(self,loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


    def forward(self, preds, target):
        # print (preds.shape)
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self._reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self._linear_combination(loss/n, nll, self.epsilon)


class LabelSmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        target = target.float()
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        # print(target)
        loss = self.bce_with_logits(pred, target)
        return loss.mean()

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        pt = torch.exp(-bce_loss)  # pt = p if label=1 else 1-p
        
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss


        return focal_loss


@HEADS.register_module()
class InfarClassification_Head(nn.Module):
    def __init__(self,):
        super(InfarClassification_Head, self).__init__()
        # TODO: 定制Head模型
        # self.conv_bin = nn.Conv3d(in_channels, 1, 1)
        # self.conv_tumor = nn.Conv3d(in_channels, 1, 1)
        # self.conv_abdomen = nn.Conv3d(in_channels, 5, 1)
        # self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.loss_bce_func = LabelSmoothBCEWithLogitsLoss()
        # self.loss_ce_func = torch.nn.CrossEntropyLoss(reduce=False)
        # self.loss_mse_func = torch.nn.SmoothL1Loss(reduction="none", beta=1)
        # self.loss_mse_func = torch.nn.MSELoss(reduction="none")
        # self.loss_labelsm_ce_func = LabelSmoothingCrossEntropy(reduction='none')
        # self._show_count = 0
        # self.avg_conv = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1, bias=False)
        # self.avg_conv.weight.data = torch.ones_like(self.avg_conv.weight) / 3

    def forward(self, inputs):
        pass
        # TODO: 定制forward网络
        # inputs = F.interpolate(inputs, scale_factor=(2.0,) * 3, mode="trilinear")
        # pred_bin = self.conv_bin(inputs)
        # pred_tumor = self.conv_tumor(inputs)
        # pred_abdomen = self.conv_abdomen(inputs)
        # return pred_bin, pred_tumor, pred_abdomen

    def loss(self, inputs, targets):

        with torch.no_grad():
            targets = targets.view(-1, 1)
            # times = times.view(-1, 1)
        
        # outs_cls, outs_reg, deep_sup_cls, deep_sup_reg = inputs[0], inputs[1], inputs[2], inputs[3]
        outs_cls, deep_sup_cls = inputs[0], inputs[1]
        loss_cls = self.loss_bce_func(outs_cls.float(), targets.float())
        print(torch.sigmoid(outs_cls).float(), targets.float())
        loss_cls = loss_cls.mean()
        # loss_reg = self.loss_mse_func(outs_reg.float(), times.float()) * 0.1
        # loss_reg = loss_reg.mean()
        loss_cls_deep_sup = self.loss_bce_func(deep_sup_cls.float(), targets.float()) * 0.5
        loss_cls_deep_sup = loss_cls_deep_sup.mean()

        # loss_cls_deep_sup_flow = self.loss_bce_func(deep_sup_cls_flow.float(), targets.float()) * 0.5
        # loss_cls_deep_sup_flow = loss_cls_deep_sup_flow.mean()
        # loss_reg_deep_sup = self.loss_mse_func(deep_sup_reg.float(), times.float()) * 0.05
        # loss_reg_deep_sup = loss_reg_deep_sup.mean()
        return {
                "loss_ce": loss_cls,
                # "loss_mse": loss_reg,
                "loss_cls_deep_sup": loss_cls_deep_sup,
                # "loss_cls_deep_sup_flow": loss_cls_deep_sup_flow,
                }
    
    def _smooth_tgt(self, tgt, iter=3):
        tgt = tgt[None, None, :, :, :]
        for _ in range(iter):
            tgt = self.avg_conv(tgt)
        scale_ratio = 1 / tgt.max()
        tgt = torch.clamp(tgt, 0, 1)[0, 0] * scale_ratio
        return tgt
    

@HEADS.register_module()
class CineClassification_Head(nn.Module):
    def __init__(self,):
        super(CineClassification_Head, self).__init__()
        # TODO: 定制Head模型
        # self.conv_bin = nn.Conv3d(in_channels, 1, 1)
        # self.conv_tumor = nn.Conv3d(in_channels, 1, 1)
        # self.conv_abdomen = nn.Conv3d(in_channels, 5, 1)
        # self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        # self.loss_bce_func = LabelSmoothBCEWithLogitsLoss()
        self.loss_ce_func = torch.nn.CrossEntropyLoss(reduce=False)
        # self.loss_mse_func = torch.nn.SmoothL1Loss(reduction="none", beta=1)
        # self.loss_mse_func = torch.nn.MSELoss(reduction="none")
        # self.loss_labelsm_ce_func = LabelSmoothingCrossEntropy(reduction='none')
        # self._show_count = 0
        # self.avg_conv = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1, bias=False)
        # self.avg_conv.weight.data = torch.ones_like(self.avg_conv.weight) / 3

    def forward(self, inputs):
        pass
        # TODO: 定制forward网络
        # inputs = F.interpolate(inputs, scale_factor=(2.0,) * 3, mode="trilinear")
        # pred_bin = self.conv_bin(inputs)
        # pred_tumor = self.conv_tumor(inputs)
        # pred_abdomen = self.conv_abdomen(inputs)
        # return pred_bin, pred_tumor, pred_abdomen

    def loss(self, inputs, targets):
        
        outs_cls, deep_sup_cls = inputs[0], inputs[1]
        loss_cls = self.loss_ce_func(outs_cls, targets)
        loss_cls = loss_cls.mean()
        loss_cls_deep_sup = self.loss_ce_func(deep_sup_cls, targets) * 0.5
        loss_cls_deep_sup = loss_cls_deep_sup.mean()

        return {
                "loss_ce": loss_cls,
                "loss_cls_deep_sup": loss_cls_deep_sup,
                }


@HEADS.register_module()
class LGEClassification_Head(nn.Module):
    def __init__(self,):
        super(LGEClassification_Head, self).__init__()
        # TODO: 定制Head模型
        # self.conv_bin = nn.Conv3d(in_channels, 1, 1)
        # self.conv_tumor = nn.Conv3d(in_channels, 1, 1)
        # self.conv_abdomen = nn.Conv3d(in_channels, 5, 1)
        # self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        # self.loss_bce_func = LabelSmoothBCEWithLogitsLoss()
        self.loss_ce_func = torch.nn.CrossEntropyLoss(reduce=False)
        # self.loss_mse_func = torch.nn.SmoothL1Loss(reduction="none", beta=1)
        # self.loss_mse_func = torch.nn.MSELoss(reduction="none")
        # self.loss_labelsm_ce_func = LabelSmoothingCrossEntropy(reduction='none')
        # self._show_count = 0
        # self.avg_conv = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1, bias=False)
        # self.avg_conv.weight.data = torch.ones_like(self.avg_conv.weight) / 3

    def forward(self, inputs):
        pass
        # TODO: 定制forward网络
        # inputs = F.interpolate(inputs, scale_factor=(2.0,) * 3, mode="trilinear")
        # pred_bin = self.conv_bin(inputs)
        # pred_tumor = self.conv_tumor(inputs)
        # pred_abdomen = self.conv_abdomen(inputs)
        # return pred_bin, pred_tumor, pred_abdomen

    def loss(self, inputs, targets):
        
        outs_cls, deep_sup_cls = inputs[0], inputs[1]
        loss_cls = self.loss_ce_func(outs_cls, targets)
        loss_cls = loss_cls.mean()
        loss_cls_deep_sup = self.loss_ce_func(deep_sup_cls, targets) * 0.5
        loss_cls_deep_sup = loss_cls_deep_sup.mean()

        return {
                "loss_ce": loss_cls,
                "loss_cls_deep_sup": loss_cls_deep_sup,
                }