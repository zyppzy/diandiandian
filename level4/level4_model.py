"""
U-Net网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次 (Conv3x3 + BN + ReLU)，U-Net 的基本卷积单元。"""

    def __init__(self, in_channals, out_channals, mid_channals=None):
        super().__init__()
        mid_channals = mid_channals or out_channals
        self.conv = nn.Sequential(
            nn.Conv2d(in_channals, mid_channals, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channals),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channals, out_channals, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channals),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    """下采样：MaxPool 减半分辨率 + DoubleConv 增加通道数。"""

    def __init__(self, in_channals, out_channals):
        super().__init__()
        self.MaxPool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channals, out_channals)

    def forward(self, x):
        x = self.MaxPool(x)
        x = self.conv(x)
        return x


class up(nn.Module):
    """上采样：恢复分辨率，并与编码器对应层做跳跃连接。"""

    def __init__(self, in_channals, out_channals, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channals, out_channals, in_channals // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channals, in_channals // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channals, out_channals)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x3 = torch.cat([x2, x1], dim=1)
        return self.conv(x3)


class ChannalAttention(nn.Module):
    """通道注意力：对不同通道加权，聚焦区分手写与印刷的关键特征。"""

    def __init__(self, channals, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channals, channals // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channals // reduction, channals, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        Avgpool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        Maxpool = F.adaptive_max_pool2d(x, 1).view(b, c)
        weight = torch.sigmoid(self.mlp(Avgpool) + self.mlp(Maxpool)).view(b, c, 1, 1)
        return x * weight


class UNet(nn.Module):
    """U-Net，单输出干净图。"""

    def __init__(self, in_channals=1, out_channals=1, base_channals=64, bilinear=True, use_attention=True):
        super().__init__()
        self.inc = DoubleConv(in_channals, base_channals)
        self.d1 = down(base_channals, base_channals * 2)
        self.d2 = down(base_channals * 2, base_channals * 4)
        self.d3 = down(base_channals * 4, base_channals * 8)
        factor = 2 if bilinear else 1
        self.d4 = down(base_channals * 8, base_channals * 16 // factor)

        bottleneck_channals = base_channals * 16 // factor
        self.attention = ChannalAttention(bottleneck_channals) if use_attention else nn.Identity()

        self.u1 = up(base_channals * 16, base_channals * 8 // factor, bilinear)
        self.u2 = up(base_channals * 8, base_channals * 4 // factor, bilinear)
        self.u3 = up(base_channals * 4, base_channals * 2 // factor, bilinear)
        self.u4 = up(base_channals * 2, base_channals, bilinear)
        self.out = nn.Conv2d(base_channals, out_channals, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x5 = self.attention(x5)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)
