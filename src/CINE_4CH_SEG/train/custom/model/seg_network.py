import numpy as np
import torch
import torch.nn as nn
import os
import sys
from custom.model.utils import build_backbone, build_head
from custom.model.registry import NETWORKS
from custom.dataset.utils import build_pipelines


@NETWORKS.register_module()
class Color3D_Network(nn.Module):

    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(Color3D_Network, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()
        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.loss_bce_func = torch.nn.BCEWithLogitsLoss(reduce=False)
    @torch.jit.ignore
    def forward(self, img, mask):
        # import time
        # s = str(time.time())
        # import SimpleITK as sitk
        # sitk.WriteImage(sitk.GetImageFromArray(img[0][0].cpu().float().numpy()), "./"+ s + "lr.nii.gz")
        outs = self.backbone(img)
        head_outs = self.head(outs)
        loss = self.head.loss(head_outs, mask)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        outs = self.backbone(img)
        head_out_lr, head_out_l, head_out_r,_ = self.head(outs)
        bin_mask = img[:, 1:, :].cpu()[0]
        head_out_lr = torch.argmax(head_out_lr, dim=1)
        head_out_lr += 1
        head_out_lr = head_out_lr.cpu()[0]        
        head_out_lr = head_out_lr * bin_mask

        # head_out_l = torch.argmax(head_out_l, dim=1)
        # head_out_l += 1
        # head_out_l = head_out_l.cpu()[0]
        # left_mask = head_out_lr.clone()
        # left_mask[left_mask > 1] = 0
        # head_out_l = head_out_l * left_mask
        # head_out_l[head_out_l == 3] = 6

        # head_out_r = torch.argmax(head_out_r, dim=1).cpu()[0]
        # head_out_r += 1
        # right_mask = head_out_lr.clone()
        # right_mask[right_mask == 1] = 0
        # right_mask[right_mask == 2] = 1
        # head_out_r = head_out_r * right_mask
        # head_out_r[head_out_r == 1] = 3
        # head_out_r[head_out_r == 2] = 4
        # head_out_r[head_out_r == 3] = 5

        # res = head_out_l + head_out_r
        # print(torch.unique(head_out_l),torch.unique(head_out_r),torch.unique(head_out_a))

        return head_out_lr

    def single_test(self, img, mask):

        outs = self.backbone(img)
        inputs = self.head(outs)

        with torch.no_grad():
            target, target1, weightgenbu, _ = mask[:,0][None], mask[:,1][None], mask[:,2][None], mask[:,3][None]

            target = target.long()

            target_1 = torch.where(target1==2, 1, 0).float()
            color_bin1 = (target > 0).float() * (weightgenbu > 0).float()
            color_bin_sum1 = torch.sum(color_bin1) + 1

            target_2 = torch.where(target1==3, 1, 0).float()
            target_3 = torch.where(target1==4, 1, 0).float()

        loss_1 = self.loss_bce_func(inputs[0], target_1)#分大类
        loss_1 = torch.sum(loss_1 * color_bin1) / color_bin_sum1

        loss_2 = self.loss_bce_func(inputs[1], target_2)#分大类
        loss_2 = torch.sum(loss_2 * color_bin1) / color_bin_sum1

        loss_3 = self.loss_bce_func(inputs[2], target_3)#分大类
        loss_3 = torch.sum(loss_3 * color_bin1) / color_bin_sum1

        return (loss_2 + loss_3 + loss_1).detach().cpu().numpy()

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

@NETWORKS.register_module()
class Seg_Network_Heart(nn.Module):
    def __init__(
        self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None
    ):
        super(Seg_Network_Heart, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, img, mask):

        with torch.no_grad():
            # 数据pipeline(augmentation)处理
            data = {"img": img, "mask": mask}
            data = self._pipeline(data)
            img, mask = data["img"], data["mask"]
            img = img.detach()
            mask = mask.detach()

        # ############## debug ##############
        # import SimpleITK as sitk
        # import time

        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(img[0, 0].detach().cpu().float().numpy()),
        #     f"/home/ltiecheng/Solutions/seg_liver/train/debug/{name}_vol.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(liver[0, 0].detach().cpu().float().numpy()),
        #     f"/home/ltiecheng/Solutions/seg_liver/train/debug/{name}_mask-seg.nii.gz",
        # )
        # raise
        # ############## debug ##############

        outs = self.backbone(img)
        head_outs = self.head(outs)
        # ############## debug ##############
        # import SimpleITK as sitk
        # import time
        # import os
        # pred_bin = torch.sigmoid(head_outs)
        # os.makedirs('./unetattdebug',exist_ok=True)
        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(pred_bin[0, 0].detach().cpu().float().numpy()), './unetattdebug/' + str(name) + "-seg.nii.gz")
        loss = self.head.loss(head_outs, mask)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        # TODO: 根据需求适配，python custom/utils/save_torchscript.py保存静态图时使用
        # img_other = torch.clamp(img, min=self._other_win_range[0], max=self._other_win_range[1])
        # img_other -= self._other_win_range[0]
        # img_other /= self._other_win_range[1] - self._other_win_range[0]
        # img = torch.cat([img, img_other], dim=1)
        outs = self.backbone(img)
        pred_bin = self.head(outs)
        pred_bin = torch.sigmoid(pred_bin)
        # pred_tumor = torch.sigmoid(pred_tumor)
        # pred_abdomen = torch.softmax(pred_abdomen, dim=1)
        return pred_bin

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

@NETWORKS.register_module()
class SegDY_Network_Heart(nn.Module):
    def __init__(
        self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None
    ):
        super(SegDY_Network_Heart, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, img, mask):

        with torch.no_grad():
            # 数据pipeline(augmentation)处理
            data = {"img": img, "mask": mask}
            data = self._pipeline(data)
            img, mask = data["img"], data["mask"]
            img = img.detach()
            mask = mask.detach()

        # ############## debug ##############
        # import SimpleITK as sitk
        # import time

        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(img[0, 0].detach().cpu().float().numpy()),
        #     f"/home/ltiecheng/Solutions/seg_liver/train/debug/{name}_vol.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(liver[0, 0].detach().cpu().float().numpy()),
        #     f"/home/ltiecheng/Solutions/seg_liver/train/debug/{name}_mask-seg.nii.gz",
        # )
        # raise
        # ############## debug ##############

        outs = self.backbone(img)
        head_outs = self.head(outs)
        # ############## debug ##############
        # import SimpleITK as sitk
        # import time
        # import os
        # pred_bin = torch.sigmoid(head_outs)
        # os.makedirs('./unetattdebug',exist_ok=True)
        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(pred_bin[0, 0].detach().cpu().float().numpy()), './unetattdebug/' + str(name) + "-seg.nii.gz")
        loss = self.head.loss(head_outs, mask)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        # TODO: 根据需求适配，python custom/utils/save_torchscript.py保存静态图时使用
        outs = self.backbone(img)
        pred_bin = self.head(outs)
        pred_bin = torch.sigmoid(pred_bin)
        # pred_tumor = torch.sigmoid(pred_tumor)
        # pred_abdomen = torch.softmax(pred_abdomen, dim=1)
        return pred_bin

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)



