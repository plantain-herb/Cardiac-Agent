import torch
import torch.nn as nn
from custom.model.registry import BACKBONES
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class CoxRegression(nn.Module):
    def __init__(self, n_in):
        # n_in 特征的维度
        super(CoxRegression, self).__init__()
        self.W = nn.Parameter(torch.zeros(n_in, 1, dtype=torch.float32))
        
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        theta = torch.matmul(x, self.W).squeeze()  
        return theta
    
    def negative_log_likelihood(self, theta, R_batch, ystatus_batch):
        exp_theta = torch.exp(theta)
        log_risk = torch.log(torch.sum(exp_theta * R_batch, dim=1))
        nll = -torch.mean((theta - log_risk) * ystatus_batch)
        return nll
    
    def evalNewData(self, test_data):
        if isinstance(test_data, list):
            test_data = torch.cat(test_data, dim=1)
        return torch.matmul(test_data, self.W)   

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

def conv3x3_2d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
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

def conv1x1_2d(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        # self.dropout = nn.Dropout3d(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.dropout(out)

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

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2d, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.dropout = nn.Dropout3d(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.dropout(out)

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

def make_res_layer2d(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(conv1x1_2d(inplanes, planes, stride), nn.BatchNorm3d(planes),)

    layers = []
    layers.append(BasicBlock2d(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock2d(planes, planes))

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

class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
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


@BACKBONES.register_module()
class ResUnet(nn.Module):
    def __init__(self, in_ch, channels=16, stride=2, blocks=3):
        super(ResUnet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride, kernel_size=3)
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
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())
        self.conv = nn.Sequential(nn.Conv3d(channel, 1, kernel_size=1, stride=1, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        sc = self.conv(x)
        return x*y + x*sc


@BACKBONES.register_module()
class ResUnet_enc(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(ResUnet_enc, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.SE = SELayer(channels * 8, 4)
        # self.fc4 = nn.Linear(channels * 8, 6)

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        # c4 = self.SE(c4)
        c5 = self.avgpool(c4)
        x = torch.flatten(c5, 1)
        # x = self.fc4(x)
        return x

@BACKBONES.register_module()
class ResUnet_enc2d(nn.Module):
    def __init__(self, in_ch, channels=16, blocks=3):
        super(ResUnet_enc2d, self).__init__()

        self.in_conv = DoubleConv2d(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer2d(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer2d(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer2d(channels * 4, channels * 8, blocks, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.SE = SELayer(channels * 8, 4)
        # self.fc4 = nn.Linear(channels * 8, 6)

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        # c4 = self.SE(c4)
        c5 = self.avgpool(c4)
        x = torch.flatten(c5, 1)
        # x = self.fc4(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


@BACKBONES.register_module()
class CNNTrans(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(CNNTrans, self).__init__()
        self.CNN = ResUnet_enc(in_ch=in_ch, channels=channels, blocks=blocks)
        # self.CNN_flow = ResUnet_enc(in_ch=3, channels=channels, blocks=blocks)
        self.CNN.fc = nn.Sequential(nn.Linear(channels * 8, 128))
        # self.CNN_flow.fc = nn.Sequential(nn.Linear(channels * 8, 128))
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim=128, depth=3, heads=4, dim_head=32, mlp_dim=256, dropout=0.1)

        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        # self.mlp_head_cls = nn.Sequential(
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 1)
        # )
        # self.fusion_proj = nn.Sequential(
        #     nn.Linear(256, 128),  # 256是concat后的维度
        # )

        self.deep_sup_cls = nn.Linear(128, 1)
        # self.deep_sup_cls_flow = nn.Linear(128, 1)
        # self.deep_sup_reg = nn.Linear(128, 1)


    def forward(self, img):
        # import time
        # import SimpleITK as sitk
        # s = str(time.time())
        # print(img.shape)
        # img_cur1 = img.detach()
        # sitk.WriteImage(sitk.GetImageFromArray(img_cur1[0][0].cpu().float().numpy()), "./"+ s + ".nii.gz")
        # hidden = None
        # F_CNN = []
        img_cur = img.detach()
        x = self.CNN(img_cur)
        x = self.CNN.fc(x)
        deep_sup_cls = self.deep_sup_cls(x)

        # flow_cur = flow.detach()
        # x_flow = self.CNN_flow(flow_cur)
        # x_flow = self.CNN_flow.fc(x_flow)
        # deep_sup_cls_flow = self.deep_sup_cls(x_flow)

        # deep_sup_reg = self.deep_sup_reg(x)
        # F_CNN.append(x)
        # F_CNN.append(x_flow)
        # F_CNN = torch.cat([x, x_flow], dim=1)
        # F_CNN = self.fusion_proj(F_CNN)
        # print(F_CNN.shape)
        # x = self.dropout(F_CNN.unsqueeze(1))
        x = self.transformer(x.unsqueeze(1))

        x_cls = self.mlp_head_cls(x)
        # x_reg = self.mlp_head_reg(x)
        # print(x_cls.shape, x_reg.shape, deep_sup_cls.shape, deep_sup_reg.shape)
        return x_cls, deep_sup_cls


@BACKBONES.register_module()
class CNNTrans_multi(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, num_classes=3):
        super(CNNTrans_multi, self).__init__()
        self.CNN_2ch = ResUnet_enc(in_ch=in_ch, channels=channels//2, blocks=blocks)
        self.CNN_4ch = ResUnet_enc(in_ch=in_ch, channels=channels//2, blocks=blocks)
        self.CNN_sa = ResUnet_enc(in_ch=in_ch, channels=channels, blocks=blocks)

        self.CNN_2ch.fc = nn.Sequential(nn.Linear(channels * 4, 64))
        self.CNN_4ch.fc = nn.Sequential(nn.Linear(channels * 4, 64))
        self.CNN_sa.fc = nn.Sequential(nn.Linear(channels * 8, 128))

        self.transformer = Transformer(dim=256, depth=6, heads=8, dim_head=32, mlp_dim=512, dropout=0.1)
        
        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )
    
        self.deep_sup_cls = nn.Linear(256, num_classes)
        self.deep_sup_cls_2ch = nn.Linear(64, num_classes)
        self.deep_sup_cls_4ch = nn.Linear(64, num_classes)
        self.deep_sup_cls_sa = nn.Linear(128, num_classes)

    def forward(self, vol_2ch, vol_4ch, vol_sa):

        x2 = self.CNN_2ch(vol_2ch)
        x2 = self.CNN_2ch.fc(x2)
        x4 = self.CNN_4ch(vol_4ch)
        x4 = self.CNN_4ch.fc(x4)
        xs = self.CNN_sa(vol_sa)
        xs = self.CNN_sa.fc(xs)

        x = torch.cat([x2, x4, xs], dim=1)  # 256
        deep_sup_cls = self.deep_sup_cls(x)
        deep_sup_cls_2ch = self.deep_sup_cls_2ch(x2)
        deep_sup_cls_4ch = self.deep_sup_cls_4ch(x4)
        deep_sup_cls_sa = self.deep_sup_cls_sa(xs)

        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x_cls = self.mlp_head_cls(x)

        return x_cls, deep_sup_cls, deep_sup_cls_2ch, deep_sup_cls_4ch, deep_sup_cls_sa


@BACKBONES.register_module()
class CNNTrans_multi2d(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(CNNTrans_multi2d, self).__init__()
        self.CNN = ResUnet_enc2d(in_ch=in_ch, channels=channels, blocks=blocks)
        # self.CNN_flow = ResUnet_enc(in_ch=3, channels=channels, blocks=blocks)
        self.CNN.fc = nn.Sequential(nn.Linear(channels * 8, 128))
        # self.CNN_flow.fc = nn.Sequential(nn.Linear(channels * 8, 128))
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim=128, depth=3, heads=4, dim_head=32, mlp_dim=256, dropout=0.1)

        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 4)
        )

        self.deep_sup_cls = nn.Linear(128, 4)
        # self.deep_sup_cls_flow = nn.Linear(128, 1)
        # self.deep_sup_reg = nn.Linear(128, 1)
        

    def forward(self, img):
        # import time
        # import SimpleITK as sitk
        # s = str(time.time())
        # print(img.shape)
        # img_cur1 = img.detach()
        # sitk.WriteImage(sitk.GetImageFromArray(img_cur1[0][0].cpu().float().numpy()), "./"+ s + ".nii.gz")
        # hidden = None
        # F_CNN = []
        img_cur = img.detach()
        x = self.CNN(img_cur)
        x = self.CNN.fc(x)
        deep_sup_cls = self.deep_sup_cls(x)

        # flow_cur = flow.detach()
        # x_flow = self.CNN_flow(flow_cur)
        # x_flow = self.CNN_flow.fc(x_flow)
        # deep_sup_cls_flow = self.deep_sup_cls(x_flow)

        # deep_sup_reg = self.deep_sup_reg(x)
        # F_CNN.append(x)
        # F_CNN.append(x_flow)
        # F_CNN = torch.cat([x, x_flow], dim=1)
        # F_CNN = self.fusion_proj(F_CNN)
        # print(F_CNN.shape)
        # x = self.dropout(F_CNN.unsqueeze(1))
        x = self.transformer(x.unsqueeze(1))

        x_cls = self.mlp_head_cls(x)
        # x_reg = self.mlp_head_reg(x)
        # print(x_cls.shape, x_reg.shape, deep_sup_cls.shape, deep_sup_reg.shape)
        return x_cls, deep_sup_cls
    
@BACKBONES.register_module()
class CNNTrans_cox(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(CNNTrans_cox, self).__init__()
        self.CNN = ResUnet_enc(in_ch=in_ch, channels=channels, blocks=blocks)
        # self.CNN_flow = ResUnet_enc(in_ch=3, channels=channels, blocks=blocks)
        self.CNN.fc = nn.Sequential(nn.Linear(channels * 8, 128))

        self.transformer = Transformer(dim=128, depth=3, heads=4, dim_head=32, mlp_dim=256, dropout=0.1)

        self.mlp_head_cls = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

        self.deep_sup_cls = nn.Linear(128, 1)
        self.cox_regression = CoxRegression(128)
        # self.deep_sup_cls_flow = nn.Linear(128, 1)
        # self.deep_sup_reg = nn.Linear(128, 1)


    def forward(self, img):
        # import time
        # import SimpleITK as sitk
        # s = str(time.time())
        # print(img.shape)
        # img_cur1 = img.detach()
        # sitk.WriteImage(sitk.GetImageFromArray(img_cur1[0][0].cpu().float().numpy()), "./"+ s + ".nii.gz")
        # hidden = None
        # F_CNN = []
        img_cur = img.detach()
        x = self.CNN(img_cur)
        x = self.CNN.fc(x)
        deep_sup_cls = self.deep_sup_cls(x)

        # flow_cur = flow.detach()
        # x_flow = self.CNN_flow(flow_cur)
        # x_flow = self.CNN_flow.fc(x_flow)
        # deep_sup_cls_flow = self.deep_sup_cls(x_flow)

        # deep_sup_reg = self.deep_sup_reg(x)
        # F_CNN.append(x)
        # F_CNN.append(x_flow)
        # F_CNN = torch.cat([x, x_flow], dim=1)
        # F_CNN = self.fusion_proj(F_CNN)
        # print(F_CNN.shape)
        # x = self.dropout(F_CNN.unsqueeze(1))
        x = self.transformer(x.unsqueeze(1))
        x_cox = self.cox_regression(x)

        x_cls = self.mlp_head_cls(x)
        # x_reg = self.mlp_head_reg(x)
        # print(x_cls.shape, x_reg.shape, deep_sup_cls.shape, deep_sup_reg.shape)
        return x_cls, x_cox, deep_sup_cls


if __name__ == "__main__":
    model = ResUnet(1, 1)
    print(model)
