import numpy as np
import torch
import torch.nn as nn
import os
import sys
from custom.model.utils import build_backbone, build_head
from custom.model.registry import NETWORKS
from custom.dataset.utils import build_pipelines

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



