import os
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom.model.utils import build_loss
from custom.model.registry import HEADS, LOSSES
from einops import repeat
from typing import Optional, Union, List


@HEADS.register_module()
class Gl_Heart_Head(nn.Module):

    def __init__(
        self,
        in_channels,
        outer_scale_factor,
        inner_scale_factor
    ):
        super(Gl_Heart_Head, self).__init__()
        self.outer_heart_conv = nn.Conv3d(in_channels, 1, 1)
        self.inner_heart_conv = nn.Conv3d(in_channels, 1, 1)
        self.outer_heart_conv_color = nn.Conv3d(in_channels, 7, 1)
        self.inner_heart_conv_color = nn.Conv3d(in_channels, 7, 1)
        self.outer_scale_factor = outer_scale_factor
        self.inner_scale_factor = inner_scale_factor
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    @torch.jit.ignore
    def forward(self, inputs):
        outer_feature, inner_feature = inputs
        outer_feature = F.interpolate(outer_feature, scale_factor=(float(self.outer_scale_factor), ) * 3, mode='trilinear', align_corners=False)
        outer_heart_pred = self.outer_heart_conv(outer_feature)
        outer_heart_pred_color = self.outer_heart_conv_color(outer_feature)

        inner_feature = F.interpolate(inner_feature, scale_factor=(float(self.inner_scale_factor), ) * 3, mode='trilinear', align_corners=False)
        inner_heart_pred = self.inner_heart_conv(inner_feature)
        inner_heart_pred_color = self.inner_heart_conv_color(inner_feature)

        return [outer_heart_pred, outer_heart_pred_color, inner_heart_pred, inner_heart_pred_color]

    @torch.jit.export
    def forward_test(self, inputs: List[torch.Tensor]):
        _, inner_feature = inputs[0], inputs[1]
        inner_feature = F.interpolate(inner_feature, scale_factor=(float(self.inner_scale_factor), ) * 3, mode='trilinear', align_corners=False)
        inner_heart_pred_color = self.inner_heart_conv_color(inner_feature)
        inner_heart_pred = self.inner_heart_conv(inner_feature)
        return inner_heart_pred, inner_heart_pred_color

    def loss(self, inputs, targets):
        outer_heart_pred, outer_heart_pred_color, inner_heart_pred, inner_heart_pred_color = inputs
        with torch.no_grad():
            #data_type_inner = repeat(data_type_outer, 'b -> (b s)', s=int(inner_heart_pred.shape[0] / data_type_outer.shape[0]))

            outer_mask, inner_mask = targets
            outer_mask = (outer_mask).float()
            outer_mask_bin = (outer_mask > 0).float()

            inner_mask_bin = (inner_mask > 0).float()
            inner_mask = inner_mask.view(-1, inner_mask.shape[2], inner_mask.shape[3], inner_mask.shape[4], inner_mask.shape[5])
            inner_mask = (inner_mask).float()

            outer_weight = torch.zeros_like(outer_mask)
            outer_weight[(outer_mask>=6) & (outer_mask<=7)] = 6
            outer_weight[(outer_mask==2)] = 6
            outer_weight[(outer_mask>=3) & (outer_mask<=5)] = 3
            outer_weight[(outer_mask==1)] = 3
            outer_weight_sum = torch.sum(outer_weight) + 1


            inner_weight = torch.zeros_like(inner_mask)
            inner_weight[(inner_mask>=6) & (inner_mask<=7)] = 6
            inner_weight[(inner_mask==2)] = 6
            inner_weight[(inner_mask>=3) & (inner_mask<=5)] = 3
            inner_weight[(inner_mask==1)] = 3
            inner_weight_sum = torch.sum(inner_weight) + 1

            outer_mask_av = (outer_mask > 0)
            outer_mask = outer_mask_av * (outer_mask - 1)
            outer_mask = F.one_hot(outer_mask.long(), 7)
            outer_mask = einops.rearrange(outer_mask, "b c d h w c1 -> b (c c1) d h w")
            outer_mask = outer_mask.float()

            inner_mask_av = (inner_mask > 0)
            inner_mask = inner_mask_av * (inner_mask - 1)
            inner_mask = F.one_hot(inner_mask.long(), 7)
            inner_mask = einops.rearrange(inner_mask, "b c d h w c1 -> b (c c1) d h w")
            inner_mask = inner_mask.float()
            
        loss = {}
        # outer_heart_loss = self._loss_unit(outer_heart_pred, outer_mask, name="outer_heart", weight_ratio=0.2)
        outer_heart_loss_bin = self.loss_func(outer_heart_pred, outer_mask_bin)
        outer_heart_loss = self.focal_loss(outer_heart_pred_color, outer_mask, alpha=0.25, gamma=2.0)
        outer_heart_loss = (outer_heart_loss*outer_weight).sum()/outer_weight_sum
        # loss.update(outer_heart_loss)

        # inner_heart_loss = self._loss_unit(inner_heart_pred, inner_mask, name="inner_heart", weight_ratio=1)
        # loss.update(inner_heart_loss)
        #loss.update(inner_tumor_loss)
        inner_heart_loss_bin = self.loss_func(inner_heart_pred, inner_mask_bin.squeeze(0))
        inner_heart_loss = self.focal_loss(inner_heart_pred_color, inner_mask)
        inner_heart_loss = (inner_heart_loss*inner_weight).sum()/inner_weight_sum

        return {"outer_heart_loss_bin": outer_heart_loss_bin * 1, "outer_heart_loss": outer_heart_loss * 2,
                "inner_heart_loss_bin": inner_heart_loss_bin * 5, "inner_heart_loss": inner_heart_loss * 10,
        }

    def _dice_loss(self, pred, tgt_mask, weight):
        prob = torch.sigmoid(pred)
        dice = (2 * torch.sum(prob * tgt_mask, dim=[1, 2, 3, 4]) + 1e-6) / (torch.sum(prob, dim=[1, 2, 3, 4]) + torch.sum(tgt_mask, dim=[1, 2, 3, 4]) + 1e-6)
        if weight is not None:
            dice = torch.sum(dice * weight) / (torch.sum(weight) + 1e-6)
        else:
            dice = torch.mean(dice)
        return 1 - dice

    def _loss_unit(self, pred_mask, tgt_mask, weight=None, name="", weight_ratio=1, dice_loss_flag=True):
        loss_bce = self.loss_func(pred_mask, tgt_mask)
        loss_bce = loss_bce.mean(dim=[1, 2, 3, 4])
        if weight is not None:
            loss_bce = torch.sum(loss_bce * weight) / (torch.sum(weight) + 1e-6)
        else:
            loss_bce = torch.mean(loss_bce)
        if dice_loss_flag:
            loss_dice = self._dice_loss(pred_mask, tgt_mask, weight)
            return {f"{name}_loss_bce": loss_bce * weight_ratio * 20 , f"{name}_loss_dice": loss_dice * weight_ratio  }
        else:
            return {f"{name}_loss_bce": loss_bce * weight_ratio}

        # loss_dice = self._dice_loss(pred_mask, tgt_mask, weight)
        # return {f"{name}_loss_dice":loss_dice}
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

        return loss
