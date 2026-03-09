import numpy as np
import torch
import torch.nn as nn
from custom.model.utils import build_backbone, build_head
from custom.model.registry import NETWORKS
from custom.dataset.utils import build_pipelines


@NETWORKS.register_module()
class InfarClassification_Network(nn.Module):
    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(InfarClassification_Network, self).__init__()

        # TODO: 定制网络
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        # self._other_win_range = other_win_range
        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._show_count = 0

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

        # results = {"img": vol.detach(), "liver": liver.detach(), "gt": gt}

    @torch.jit.ignore
    def forward(self, img, mask, flow, gt):
        # TODO: 定制forward网络

        aug_data = dict(img=img.squeeze(1), mask=mask.squeeze(1), others=[flow.squeeze(1)[:,0:1,:,:,:], flow.squeeze(1)[:,1:2,:,:,:]])
        aug_data = self._pipeline(aug_data)
        aug_img, aug_mask, aug_flow = aug_data["img"], aug_data["mask"], aug_data["others"]
        aug_flows = torch.cat([aug_flow[0] ,aug_flow[1]], dim=1)
        vol = torch.cat([aug_img ,aug_mask, aug_flows], dim=1)

        # vol = torch.cat([img ,mask, flow], dim=1)


        # import os
        # import SimpleITK as sitk
        # for idx in range(vol.size(0)):
        
        # si = vol[0, 0].cpu().numpy()
        # si = si.astype(np.uint8)
        # si = sitk.GetImageFromArray(si)
        # sitk.WriteImage(si, '/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/ori_data/grad_cam_output2/vol1.nii.gz')

        # 多加一类
        outs_cls, deep_sup_cls = self.backbone(vol)
        gt = gt.view(gt.shape[0] * gt.shape[1])
        # time = time.view(time.shape[0] * time.shape[1])
        outs_cls = outs_cls.view(outs_cls.shape[0] * outs_cls.shape[1], -1)
        # outs_reg = outs_reg.view(outs_reg.shape[0] * outs_reg.shape[1], -1)

        loss = self.head.loss((outs_cls, deep_sup_cls), gt)
        return loss

    @torch.jit.export
    def forward_test(self, img):

        # print(img.shape)
        outs_cls, _ = self.backbone(img)
        outs_cls = outs_cls.view(outs_cls.shape[0] * outs_cls.shape[1], -1)
        # outs_reg = outs_reg.view(outs_reg.shape[0] * outs_reg.shape[1], -1)
        
        pred = torch.sigmoid(outs_cls)
        # sofxmax for grad cam
        # pred = torch.softmax(outs_cls, dim=1)
        return pred

    def single_test(self, img, gt):
        pred_cls = self.forward_test(img)
        # print(pred_cls, pred_reg)
        # print(pred.shape, gt.shape)  # torch.Size([7, 6]) torch.Size([1, 7])
        return pred_cls, gt

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)


@NETWORKS.register_module()
class CineClassification_Network(nn.Module):
    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(CineClassification_Network, self).__init__()

        # TODO: 定制网络
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        # self._other_win_range = other_win_range
        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._show_count = 0

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, img_4ch, img_sa, img_lge, gt):
        # TODO: 定制forward网络
        aug_4ch = self._pipeline(dict(img=img_4ch.squeeze(1)))["img"]
        aug_sa  = self._pipeline(dict(img=img_sa.squeeze(1)))["img"]
        aug_lge = self._pipeline(dict(img=img_lge.squeeze(1)))["img"]

        # 多加一类
        outs_cls, deep_sup_cls,  deep_sup_cls_4ch, deep_sup_cls_sa, deep_sup_cls_lge = self.backbone(aug_4ch, aug_sa, aug_lge)
        gt = gt.view(gt.shape[0] * gt.shape[1])
        loss = self.head.loss((outs_cls, deep_sup_cls, deep_sup_cls_4ch, deep_sup_cls_sa, deep_sup_cls_lge), gt)
        return loss

    @torch.jit.export
    def forward_test(self, img_4ch, img_sa, img_lge):
        outs, _, _, _, _ = self.backbone(img_4ch, img_sa, img_lge)
        pred = torch.softmax(outs, dim=1)
        return pred

    def single_test(self, img_4ch, img_sa, img_lge, gt):
        # img = img.squeeze(1)
        img_4ch = img_4ch.squeeze(1)
        img_sa  = img_sa.squeeze(1)
        img_lge = img_lge.squeeze(1)
        pred = self.forward_test(img_4ch, img_sa, img_lge)
        return pred, gt

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

@NETWORKS.register_module()
class LGEClassification_Network(nn.Module):
    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(LGEClassification_Network, self).__init__()

        # TODO: 定制网络
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        # self._other_win_range = other_win_range
        # self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._show_count = 0

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

        # results = {"img": vol.detach(), "liver": liver.detach(), "gt": gt}

    @torch.jit.ignore
    def forward(self, img, gt):
        # TODO: 定制forward网络

        # aug_img = self._pipeline
        # aug_data = dict(img=img.squeeze(1))
        # aug_data = self._pipeline(aug_data)
        # aug_img= aug_data["img"]
        # print(aug_img.shape, aug_mask.shape, aug_flows.shape)
        # vol = aug_img
        # import os
        # import SimpleITK as sitk
        # for idx in range(img.size(0)):
        #
        #     si = img[idx, 0].cpu().numpy() * 255
        #     si = si.astype(np.uint8)
        #     si = sitk.GetImageFromArray(si)
        #     sitk.WriteImage(si, save_prefix + '-vol1.nii.gz')

        # print(img.shape, mask.shape, _img.shape, _mask.shape)
        # center loss
        # outs, deep_sup_outs, center_x = self.backbone(img)
        # gt = gt.view(gt.shape[0]*gt.shape[1])
        # outs = outs.view(outs.shape[0]*outs.shape[1], -1)
        # deep_sup_outs = deep_sup_outs.view(deep_sup_outs.shape[0]*deep_sup_outs.shape[1], -1)
        # center_x = center_x.view(center_x.shape[0] * center_x.shape[1], -1)
        # loss = self.head.loss((outs, deep_sup_outs, center_x), gt)

        # 多加一类
        outs_cls, deep_sup_cls = self.backbone(img.squeeze(1))
        gt = gt.view(gt.shape[0] * gt.shape[1]) 
        # time = time.view(time.shape[0] * time.shape[1])
        outs_cls = outs_cls.view(outs_cls.shape[0] * outs_cls.shape[1], -1)
        # outs_reg = outs_reg.view(outs_reg.shape[0] * outs_reg.shape[1], -1)
        loss = self.head.loss((outs_cls, deep_sup_cls), gt)
        return loss

    @torch.jit.export
    def forward_test(self, img):

        # print(img.shape)
        outs, deep_sup_outs = self.backbone(img)
        outs = outs.view(outs.shape[0] * outs.shape[1], -1)
        # deep_sup_outs = deep_sup_outs.view(deep_sup_outs.shape[0] * deep_sup_outs.shape[1], -1)
        # pred = torch.sigmoid(outs)
        pred = torch.softmax(outs, dim=1)
        return pred

    def single_test(self, img, gt):
        img = img.squeeze(1)
        pred = self.forward_test(img)
        # print(pred.shape, gt.shape)  # torch.Size([7, 6]) torch.Size([1, 7])
        return pred, gt

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

@NETWORKS.register_module()
class InfarCoxClassification_Network(nn.Module):
    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(InfarCoxClassification_Network, self).__init__()

        # TODO: 定制网络
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        # self._other_win_range = other_win_range
        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._show_count = 0

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

        # results = {"img": vol.detach(), "liver": liver.detach(), "gt": gt}

    @torch.jit.ignore
    def forward(self, img, mask, flow, gt, time):
        # TODO: 定制forward网络

        # aug_img = self._pipeline
        aug_data = dict(img=img.squeeze(1), mask=mask.squeeze(1), others=[flow.squeeze(1)[:,0:1,:,:,:], flow.squeeze(1)[:,1:2,:,:,:]])
        aug_data = self._pipeline(aug_data)
        aug_img, aug_mask, aug_flow = aug_data["img"], aug_data["mask"], aug_data["others"]
        aug_flows = torch.cat([aug_flow[0] ,aug_flow[1]], dim=1)
        # print(aug_img.shape, aug_mask.shape, aug_flows.shape)
        vol = torch.cat([aug_img, aug_mask, aug_flows], dim=1)
        # import os
        # import SimpleITK as sitk
        # for idx in range(img.size(0)):
        #
        #     si = img[idx, 0].cpu().numpy() * 255
        #     si = si.astype(np.uint8)
        #     si = sitk.GetImageFromArray(si)
        #     sitk.WriteImage(si, save_prefix + '-vol1.nii.gz')

        # print(img.shape, mask.shape, _img.shape, _mask.shape)
        # center loss
        # outs, deep_sup_outs, center_x = self.backbone(img)
        # gt = gt.view(gt.shape[0]*gt.shape[1])
        # outs = outs.view(outs.shape[0]*outs.shape[1], -1)
        # deep_sup_outs = deep_sup_outs.view(deep_sup_outs.shape[0]*deep_sup_outs.shape[1], -1)
        # center_x = center_x.view(center_x.shape[0] * center_x.shape[1], -1)
        # loss = self.head.loss((outs, deep_sup_outs, center_x), gt)

        # 多加一类
        outs_cls, outs_cox, deep_sup_cls = self.backbone(vol)
        gt = gt.view(gt.shape[0] * gt.shape[1])
        time = time.view(time.shape[0] * time.shape[1])
        outs_cls = outs_cls.view(outs_cls.shape[0] * outs_cls.shape[1], -1)
        outs_cox = outs_cox.view(outs_cox.shape[0], -1)
        loss = self.head.loss((outs_cls, outs_cox, deep_sup_cls), gt, time)
        return loss

    @torch.jit.export
    def forward_test(self, img):

        # print(img.shape)
        _, outs_cls, _ = self.backbone(img)
        # outs_cls = outs_cls.view(outs_cls.shape[0] * outs_cls.shape[1], -1)
        
        pred = torch.sigmoid(outs_cls)
        # sofxmax for grad cam
        # pred = torch.softmax(outs_cls, dim=1)
        return pred

    def single_test(self, img, gt):
        pred_cls = self.forward_test(img)
        # print(pred_cls, pred_reg)
        # print(pred.shape, gt.shape)  # torch.Size([7, 6]) torch.Size([1, 7])
        return pred_cls, gt

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
