import torch
import torch.nn as nn
import torch.nn.functional as F
from custom.model.utils import build_loss
from custom.model.registry import HEADS, LOSSES
import einops
@LOSSES.register_module()
class BceLoss(nn.BCEWithLogitsLoss):
    pass

@HEADS.register_module()
class HeartSegPyramid_Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super(HeartSegPyramid_Head, self).__init__()
        self.conv_bin1 = nn.Conv3d(in_channels, 1, 1)
        self.conv_bin2 = nn.Conv3d(in_channels, 1, 1)
        self.conv_bin3 = nn.Conv3d(in_channels, 1, 1)

        self.conv_heatmap1 = nn.Conv3d(in_channels, 7, 1)
        self.conv_heatmap2 = nn.Conv3d(in_channels, 7, 1)
        self.conv_heatmap3 = nn.Conv3d(in_channels, 7, 1)

        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input1, input2, input3):
        inputs = F.interpolate(input1, scale_factor=(1.0,) * 3, mode="trilinear", align_corners=False)
        pred_bin1 = self.conv_bin1(inputs)
        pred_heatmap1 = self.conv_heatmap1(inputs)

        inputs = F.interpolate(input2, scale_factor=(1.0,) * 3, mode="trilinear", align_corners=False)
        pred_bin2 = self.conv_bin2(inputs)
        pred_heatmap2 = self.conv_heatmap2(inputs)

        inputs = F.interpolate(input3, scale_factor=(1.0,) * 3, mode="trilinear", align_corners=False)
        pred_bin3 = self.conv_bin3(inputs)
        pred_heatmap3 = self.conv_heatmap3(inputs)

        return pred_bin1, pred_bin2, pred_bin3, pred_heatmap1, pred_heatmap2, pred_heatmap3

    def loss(self, inputs, targets):

        loss = []
        pred_bin1, pred_bin2, pred_bin3, pred_heatmap1, pred_heatmap2, pred_heatmap3 = inputs
        target1, target2, target3 = targets[:, 0: 1], targets[:, 1: 2], targets[:, 2: 3]
        target_bin1 = (target1 > 0).float()
        pred_bin1_loss = self.loss_func(pred_bin1, target_bin1)
        target_bin2 = (target2 > 0).float()
        pred_bin2_loss = self.loss_func(pred_bin2, target_bin2)
        target_bin3 = (target3 > 0).float()
        pred_bin3_loss = self.loss_func(pred_bin3, target_bin3)

        weight1 = torch.zeros_like(target1)
        weight1[(target1>=6) & (target1<=7)] = 6
        weight1[(target1==2)] = 6
        weight1[(target1>=3) & (target1<=5)] = 3
        weight1[(target1==1)] = 3
        weight1_sum = torch.sum(weight1) + 1

        weight2 = torch.zeros_like(target2)
        weight2[(target2>=6) & (target2<=7)] = 3
        weight2[(target2==2)] = 3
        weight2[(target2>=3) & (target2<=5)] = 1.5
        weight2[(target2==1)] = 1.5
        weight2_sum = torch.sum(weight2) + 1

        weight3 = torch.zeros_like(target3)
        weight3[(target3>=6) & (target3<=7)] = 1
        weight3[(target3==2)] = 1
        weight3[(target3>=3) & (target3<=5)] = 1.25
        weight3[(target3==1)] = 1.25
        weight3_sum = torch.sum(weight3) + 1

        # debug
        # import SimpleITK as sitk
        # import time
        # s = str(time.time())
        # for dx in range(len(target1)):
        #     sitk.WriteImage(sitk.GetImageFromArray(target1[dx, 0].detach().cpu().numpy().astype("float")), "./" + s + "_" + str(dx) + "_mask_seg-seg.nii.gz")
        #     sitk.WriteImage(sitk.GetImageFromArray((weight1[dx, 0] - 1).detach().cpu().numpy().astype("float")), "./" + s + "_" + str(dx) + "_loss_weight-seg.nii.gz")


        target1_av = (target1 > 0)
        target1 = target1_av * (target1 - 1)
        target1 = F.one_hot(target1.long(), 7)
        target1 = einops.rearrange(target1, "b c d h w c1 -> b (c c1) d h w")
        target1 = target1.float()

        target2_av = (target2 > 0)
        target2 = target2_av * (target2 - 1)
        target2 = F.one_hot(target2.long(), 7)
        target2 = einops.rearrange(target2, "b c d h w c1 -> b (c c1) d h w")
        target2 = target2.float()

        target3_av = (target3 > 0)
        target3 = target3_av * (target3 - 1)
        target3 = F.one_hot(target3.long(), 7)
        target3 = einops.rearrange(target3, "b c d h w c1 -> b (c c1) d h w")
        target3 = target3.float()

        pred_color1_loss = self.focal_loss(pred_heatmap1, target1, alpha=0.25, gamma=2.0)
        pred_color1_loss = (pred_color1_loss*weight1).sum()/weight1_sum
        pred_color2_loss = self.focal_loss(pred_heatmap2, target2, alpha=0.25, gamma=2.0)
        pred_color2_loss = (pred_color2_loss*weight2).sum()/weight2_sum
        pred_color3_loss = self.focal_loss(pred_heatmap3, target3, alpha=0.25, gamma=2.0)
        pred_color3_loss = (pred_color3_loss*weight3).sum()/weight3_sum


        return {"pred_bin1_loss": pred_bin1_loss * 1, "pred_bin2_loss": pred_bin2_loss * 1, "pred_bin3_loss": pred_bin3_loss * 10,
                "pred_color1_loss": pred_color1_loss * 2, "pred_color2_loss": pred_color2_loss * 2, "pred_color3_loss": pred_color3_loss * 20,
        }
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

        return loss
    

@HEADS.register_module()
class DYSegPyramid_Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super(DYSegPyramid_Head, self).__init__()
        self.conv_bin1_1 = nn.Conv3d(in_channels, 1, 1)
        self.conv_bin2_1 = nn.Conv3d(in_channels, 1, 1)

        self.conv_bin1_2 = nn.Conv3d(in_channels, 1, 1)
        self.conv_bin2_2 = nn.Conv3d(in_channels, 1, 1)

        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input1, input2):
        inputs = F.interpolate(input1, scale_factor=(1.0,) * 3, mode="trilinear", align_corners=False)
        pred_bin1_1 = self.conv_bin1_1(inputs)
        pred_bin1_2 = self.conv_bin1_2(inputs)

        inputs = F.interpolate(input2, scale_factor=(1.0,) * 3, mode="trilinear", align_corners=False)
        pred_bin2_1 = self.conv_bin2_1(inputs)
        pred_bin2_2 = self.conv_bin2_2(inputs)

        return pred_bin1_1, pred_bin2_1, pred_bin1_2, pred_bin2_2

    def loss(self, inputs, target1, target2):

        loss = []
        weight_ratios = [1, 10, 1, 10]
        restore_thresh = [0.25, 0.5]
        for idx, pred in enumerate(inputs[:2]):
            tgt = target1[:, idx : (idx + 1)]
            # print(torch.sum(tgt),1111)
            loss.append(
                self._loss_unit(
                    pred, tgt, "loss1_bin_" + str(idx), restore_thresh[idx]
                )
            )
        
        for idx, pred in enumerate(inputs[2:]):
            tgt = target2[:, idx : (idx + 1)]
            # debug
            # import SimpleITK as sitk
            # import time
            # s = str(time.time())
            # for dx in range(len(tgt)):
            #     sitk.WriteImage(sitk.GetImageFromArray(tgt[dx, 0].detach().cpu().numpy().astype("float")), "./" + s + "_" + str(dx) + "_tgt.nii.gz")

            # print(torch.sum(tgt),2222)
            loss.append(
                self._loss_unit(
                    pred, tgt, "loss2_bin_" + str(idx), restore_thresh[idx]
                )
            )


        ret_loss = {}
        for idx, l in enumerate(loss):
            weight_ratio = weight_ratios[idx]
            for k, v in l.items():
                ret_loss[k] = v * weight_ratio

        return ret_loss
    

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

        return loss
    
    def _loss_unit(self, inputs, targets, idx, restore_thresh):

        with torch.no_grad():
            targets = (targets > restore_thresh) * 1.0
            weight = torch.ones_like(targets)
            weight = weight + (((targets > 0).float()) * 6)

        loss = self.loss_func(inputs, targets)
  
        loss = loss * weight
        loss = torch.sum(loss) / (torch.sum(weight) + 1)
        # loss = loss.mean()

        return {"loss_{}".format(idx): loss}







