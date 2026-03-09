from distutils.command.build import build
import numpy as np
import torch
import torch.nn as nn
from custom.model.utils import build_backbone
from custom.model.registry import BACKBONES
from einops import repeat
from typing import Optional, Union, List

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(conv1x1(inplanes, planes, stride), nn.BatchNorm3d(planes),)

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class ResUnet(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(ResUnet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)

    def forward(self, input):
        c1 = self.in_conv(input)
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



@BACKBONES.register_module()
class MSnet_Heart(nn.Module):

    def __init__(self, outer_backbone, inner_backbone):
        super(MSnet_Heart, self).__init__()
        self.outer_backbone = build_backbone(outer_backbone)
        self.inner_backbone = build_backbone(inner_backbone)

    def forward(self, outer_img, inner_img, inner_grids, outer_feature: Optional[List[torch.Tensor]]=None):
        # outer_img: B * C * D * H * W
        # inner_img: B * S * C * D * H * W
        # inner_grids: B * S * D * H * W * 3

        feature_pyramid = []
        if outer_feature is None:
            outer_feature = self.outer_backbone(outer_img)
        feature_pyramid.append(outer_feature[-1])

        B, S, C, D, H, W = inner_img.shape
        B_grid, S_grid, D_grid, H_grid, W_grid, _ = inner_grids.shape
        inner_img = inner_img.view(-1, C, D, H, W)
        inner_grids = inner_grids.view(-1, D_grid, H_grid, W_grid, 3)
        sampled_features = []
        for idx in range(len(outer_feature)):
            #outer_feature_reshape = repeat(outer_feature[idx], 'b c d w h -> (b s) c d w h', s=S)
            outer_feature_reshape = outer_feature[idx].repeat(S, 1, 1, 1, 1) 
            sampled_feature = torch.nn.functional.grid_sample(outer_feature_reshape, inner_grids, align_corners=True)
            sampled_features.append(sampled_feature)
        sampled_features = torch.cat(sampled_features, dim=1)

        inner_feature = self.inner_backbone(inner_img, sampled_features)
        feature_pyramid.append(inner_feature)

        return feature_pyramid, outer_feature


@BACKBONES.register_module()
class ResMSUnet_Heart(ResUnet):

    def __init__(self, in_ch, extra_in_ch, channels=16, blocks=3):
        super(ResMSUnet_Heart, self).__init__(in_ch, channels, blocks)

        self.layer1 = make_res_layer(channels + extra_in_ch, channels * 2, blocks, stride=2)
        self.conv7 = DoubleConv(channels * 3 + extra_in_ch, channels)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, input, extra_feature: torch.Tensor):
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


#@BACKBONES.register_module()
# class ResBaseMSUnet(ResUnet):

#     def __init__(self, in_ch, channels=16, blocks=3):
#         super(ResBaseMSUnet, self).__init__(in_ch, channels, blocks)
#         self.in_conv = DoubleConv(in_ch, channels, stride=4, kernel_size=7)
#         self.base = 
#     def forward(self, input):
#         return [super(ResBaseMSUnet, self).forward(input)]



@BACKBONES.register_module()
class ResBaseMSUnet_v1_Heart(ResUnet):

    def __init__(self, in_ch, channels=16, blocks=3):
        super(ResBaseMSUnet_v1_Heart, self).__init__(channels // 2, channels, blocks)
        self._pre_conv = DoubleConv(in_ch, channels // 2, stride=2, kernel_size=3)
        self.in_conv = DoubleConv(channels // 2, channels, stride=2, kernel_size=3)
        self.base = ResUnet(channels // 2, channels, blocks)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels * 3 // 2, channels)

    def forward(self, input):
        inputs = self._pre_conv(input)
        c7 = self.base.forward(inputs)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, inputs], dim=1)
        c8 = self.conv8(merge8)

        return [c8]


#@BACKBONES.register_module()
# class ResBaseMSUnet_v2(ResUnet):

#     def __init__(self, in_ch, channels=16, blocks=3):
#         super(ResBaseMSUnet_v2, self).__init__(channels // 2, channels, blocks)
#         self._pre_conv = DoubleConv(in_ch, channels // 2, stride=2, kernel_size=3)
#         self.in_conv = DoubleConv(channels // 2, channels, stride=2, kernel_size=3)

#         self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv8 = DoubleConv(channels * 3 // 2, channels)

#     def forward(self, input):
#         inputs = self._pre_conv(input)
#         c7 = super(ResBaseMSUnet_v2, self).forward(inputs)

#         up_8 = self.up8(c7)
#         merge8 = torch.cat([up_8, inputs], dim=1)
#         c8 = self.conv8(merge8)

#         return [c7, c8]
