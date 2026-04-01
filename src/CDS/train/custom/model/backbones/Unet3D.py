import torch
import torch.nn as nn
from custom.model.registry import BACKBONES

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
        padding_mode="reflect",
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode="reflect")


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


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


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


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 1, padding_mode="zeros"),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
        )

    def forward(self, input):
        return self.conv(input)


@BACKBONES.register_module()
class Unet3D(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(Unet3D, self).__init__()
        self.conv1 = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.conv2 = DoubleConv(channels, channels * 2)
        self.conv3 = DoubleConv(channels * 2, channels * 4)
        self.conv4 = DoubleConv(channels * 4, channels * 8)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.up5 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)

    def forward(self, input):
        c1 = self.conv1(input)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

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
class Unet3D_ori(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(Unet3D_ori, self).__init__()
        self.conv1 = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.conv2 = DoubleConv(channels, channels * 2)
        self.conv3 = DoubleConv(channels * 2, channels * 4)
        self.conv4 = DoubleConv(channels * 4, channels * 8)
        self.conv5 = DoubleConv(channels * 8, channels * 16)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.up6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv6 = DoubleConv(channels * 24, channels * 8)
        self.up7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv7 = DoubleConv(channels * 12, channels * 4)
        self.up8 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv8 = DoubleConv(channels * 6, channels * 2)
        self.up9 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv9 = DoubleConv(channels * 3, channels)

    def forward(self, input):
        c1 = self.conv1(input)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        return c9

class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class GridAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, sub_sample_factor=(2,2,2), mode='concatenation'):
        super(GridAttentionBlock3D, self).__init__()

        # Default parameter set
        self.mode = mode
        self.sub_sample_kernel_size = sub_sample_factor
        self.upsample_mode = 'trilinear'
        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # Output transform
        self.W = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=2, padding=0, bias=False)
        self.phi = nn.Conv3d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x) # size 减少一半
        theta_x_size = theta_x.size()  

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = nn.functional.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=False)
        f = nn.functional.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = nn.functional.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, y
@BACKBONES.register_module()
class Unet3D_att(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(Unet3D_att, self).__init__()
        self.conv1 = DoubleConv(in_ch, channels, stride=2, kernel_size=3) 
        self.conv2 = DoubleConv(channels, channels * 2)
        self.conv3 = DoubleConv(channels * 2, channels * 4)
        self.conv4 = DoubleConv(channels * 4, channels * 8)
        self.conv5 = DoubleConv(channels * 8, channels * 16)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.up6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv6 = DoubleConv(channels * 24, channels * 8)
        self.up7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv7 = DoubleConv(channels * 12, channels * 4)
        self.up8 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv8 = DoubleConv(channels * 6, channels * 2)
        self.up9 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv9 = DoubleConv(channels * 3, channels)

        self.gating = UnetGridGatingSignal3(channels * 16, channels * 8, kernel_size=(1, 1, 1), is_batchnorm=True)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=channels * 2, gating_channels=channels * 8,
                                                    inter_channels=channels * 2, sub_sample_factor=(2,2,2))
        self.attentionblock3 = GridAttentionBlock3D(in_channels=channels * 4, gating_channels=channels * 8,
                                                    inter_channels=channels * 4, sub_sample_factor=(2,2,2))
        self.attentionblock4 = GridAttentionBlock3D(in_channels=channels * 8, gating_channels=channels * 8,
                                                    inter_channels=channels * 8, sub_sample_factor=(2,2,2))

    def forward(self, input):
        c1 = self.conv1(input)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        
        gating = self.gating(c5)

        g_conv4, att4 = self.attentionblock4(c4, gating)
        g_conv3, att3 = self.attentionblock3(c3, gating)
        g_conv2, att2 = self.attentionblock2(c2, gating)
    
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, att4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, att3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, att2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        return c9

if __name__ == "__main__":
    model = Unet3D_att(1, 1)
    print(model)
