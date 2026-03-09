import numpy as np
import torch
import torch.nn as nn
from custom.model.utils import build_backbone
from custom.model.registry import BACKBONES
from custom.model.backbones.ResUnet import DoubleConv, ResUnet, make_res_layer


@BACKBONES.register_module()
class SCnet(nn.Module):

    def __init__(self, in_ch, data_pyramid_level, data_pyramid_step, inner_backbone):
        super(SCnet, self).__init__()
        self._in_ch = in_ch
        self._data_pyramid_level = data_pyramid_level
        self._data_pyramid_step = data_pyramid_step
        assert len(inner_backbone) == self._data_pyramid_level
        self.backbone_base = build_backbone(inner_backbone[0])
        self.backbone_other0 = build_backbone(inner_backbone[1])
        self.backbone_other1 = build_backbone(inner_backbone[2])

    def forward(self, input):
        feature_pyramid = []
        pre_feature = self.backbone_base(input[:, :self._in_ch])
        feature_pyramid.append(pre_feature)

        level = 0
        in_data = input[:, (self._in_ch * (level + 1)):(self._in_ch * (level + 2))]
        sampled_feature = self._sample_crop_feature(pre_feature)
        pre_feature = self.backbone_other0(in_data, sampled_feature)
        feature_pyramid.append(pre_feature)

        level = 1
        in_data = input[:, (self._in_ch * (level + 1)):(self._in_ch * (level + 2))]
        sampled_feature = self._sample_crop_feature(pre_feature)
        pre_feature = self.backbone_other1(in_data, sampled_feature)
        feature_pyramid.append(pre_feature)

        return feature_pyramid

    def _sample_crop_feature(self, feature):
        feature_shape = feature.size()
        b, c, d, h, w = feature_shape
        with torch.no_grad():
            src_shape = torch.tensor(feature_shape[2:], device=feature.device)
            grid = []
            for cent_px, ts, ps_half in zip([d // 2, h // 2, w // 2], (0.5, ) * 3, [d // 2, h // 2, w // 2]):
                p_s = cent_px - ps_half * ts
                p_e = cent_px + ps_half * ts - (ts / 2)
                grid.append(torch.arange(p_s, p_e, ts, device=feature.device))
            grid = torch.meshgrid(grid[0], grid[1], grid[2])
            grid = [g[:, :, :, None] for g in grid]  # shape (d,h,w,(zyx))
            grid = torch.cat(grid, dim=3)
            grid *= 2
            grid /= (src_shape - 1)[None, None, None, :]
            grid -= 1
            grid = grid.flip(dims=[3])[None]
            grid = grid.expand((b, d, h, w, 3))
            grid = grid.detach()
        feature = torch.nn.functional.grid_sample(feature, grid, align_corners=True)
        return feature
    
@BACKBONES.register_module()
class SCnet_2_1c(nn.Module):
    def __init__(self, in_ch, data_pyramid_level, data_pyramid_step, inner_backbone):
        super(SCnet_2_1c, self).__init__()
        self._in_ch = in_ch
        self._data_pyramid_level = data_pyramid_level
        self._data_pyramid_step = data_pyramid_step
        assert len(inner_backbone) == self._data_pyramid_level
        self.backbone_base = build_backbone(inner_backbone[0])
        self.backbone_other0 = build_backbone(inner_backbone[1])

    def forward(self, input):
        feature_pyramid = []
        # with torch.no_grad():
        img = input[:, :self._in_ch]
        pre_feature = self.backbone_base(img)
        feature_pyramid.append(pre_feature)

        level = 0
        # with torch.no_grad():
        img = input[:, (self._in_ch * (level + 1)): (self._in_ch * (level + 2))]

        sampled_feature = self._sample_crop_feature(pre_feature)
        pre_feature = self.backbone_other0(img, sampled_feature)
        feature_pyramid.append(pre_feature)

        return feature_pyramid

    def _sample_crop_feature(self, feature):
        feature_shape = feature.size()
        b, c, d, h, w = feature_shape
        # with torch.no_grad():
        src_shape = torch.tensor(feature_shape[2:], device=feature.device)
        grid = []
        for cent_px, ts, ps_half in zip([d // 2, h // 2, w // 2], (0.5,) * 3, [d // 2, h // 2, w // 2]):
            p_s = cent_px - ps_half * ts
            p_e = cent_px + ps_half * ts - (ts / 2)
            grid.append(torch.arange(p_s, p_e, ts, device=feature.device))
        grid = torch.meshgrid(grid[0], grid[1], grid[2])
        grid = [g[:, :, :, None] for g in grid]  # shape (d,h,w,(zyx))
        grid = torch.cat(grid, dim=3)
        grid *= 2
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        grid = grid.flip(dims=[3])[None]
        grid = grid.expand((b, d, h, w, 3))
        grid = grid.detach()
        feature = torch.nn.functional.grid_sample(feature, grid, align_corners=True)
        return feature


@BACKBONES.register_module()
class ResSCUnet(ResUnet):

    def __init__(self, in_ch, extra_in_ch, channels=16, stride=2, blocks=3):
        super(ResSCUnet, self).__init__(in_ch, channels, stride, blocks)

        self.layer1 = make_res_layer(channels + extra_in_ch, channels * 2, blocks, stride=2)
        self.conv7 = DoubleConv(channels * 3 + extra_in_ch, channels)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, input, extra_feature):
        c1 = self.in_conv(input)
        c1 = torch.cat([c1, extra_feature], dim=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        return c7