import os
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DoubleConv(nn.model):
    #两次卷积-归一处理-激活，U-Net的基本单元
    def __init__(self,in_channals,out_channals,mid_channals=None):
        super().__init__()
        mid_channals=mid_channals or out_channals
        self.conv=nn.Sequential(
            nn.Conv2d(in_channals,mid_channals,stride=3,padding=1,bias=False),
            nn.BatchNorm2d(mid_channals),
            nn.ReLU(),
            nn.Conv2d(mid_channals,out_channals,stride=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channals),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

#下采样,减半分辨率，增加通道数
class down(nn.Module):
    def __init__(self,in_channals,out_channals):
        super().__init__()
        self.MaxPool=nn.MaxPool2d(2),
        self.conv=DoubleConv(in_channals,out_channals)

    def forward(self,x):
        x=self.MaxPool(x)
        x=self.comv(x)
        return x
#上采样，回复图像，与编码器对应层做跳跃链接
class up(nn.Module):
    def __init__(self,in_channals,out_channals,bilinear=True):
        #what is bilinear  bilinear是布尔参数，选择上采样的方式，这里是双线性插值&卷积
        #双线性插值：根据周围四个像素值加权算出新的像素值，没有学习参数，把图像放大两倍
        #所以在后面加入一个卷积，保存学习参数，修正和融合性信息
        #bilinear为False时，是有参数可学习的上采样，放大尺寸+改变通道数
        super().__init__()
        if bilinear:
            #双线性上采样??+卷积，参数少
            self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
            self.conv=DoubleConv(in_channals,out_channals,in_channals//2)
        else:
            # 转置卷积上采样
            self.up=nn.ConvTranspose2d(in_channals, in_channals//2, kernel_size=2, stride=2)
            self.conv=DoubleConv(in_channals, out_channals)

    def forward(self,x1,x2):
        #x2是编码器跳跃链接的特征，x1是解码器上采样之后的结果
        x1=self.up(x1)#上采样
        dy=x2.size(2)-x1.size(2)#高度差
        dx=x2.size(3)-x1.size(3)#宽度差
        x1=F.pad(x1,[dx//2,dx-dx//2,dy//2,dy-dy//2])#补齐尺寸??
        x3=torch.cat([x2,x1],dim=1)#拼接两个结果，通道数翻倍，这里只是简单的相加
        return self.conv(x3)#再做一次卷积，“融合”，把通道数降下来，输出结果

class ChannalAttention(nn.Module):
    #集中注意力，聚焦区分手写和印刷的关键特征通道
    #对不同通道加权，用maxpool和avgpool两种统计量，经mlp融合生成通道权重
    def __init__(self,channals,reduction=16):
        #压缩比=16，先压缩再扩张，减少参数量，增加学习效率
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(channals,channals//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(channals//reduction,channals,bias=False)
        )

    def forward(self,x):
        b,c,_,_=x.size()
        Avgpool=F.adaptive_avg_pool2d(x, 1).view(b, c)#平均池化看普遍情况
        Maxpool=F.adaptive_max_pool2d(x, 1).view(b, c)#最大池化注重突出特征
        #压缩到两个维度，方便经过全连接层
        #用mlp学习各个通道的重要性
        weight=torch.sigmoid(self.mlp(Avgpool)+self.mlp(Maxpool)).view(b,c,1,1)
        #再扩张回原来维度
        return x*weight

class UNet(nn.Module):
    def __init__(self,in_channals=1,out_channals=2,base_channals=64,bilinear=True,use_attention=True):
        super().__init__()
        self.inc = DoubleConv(in_channals, base_channals)
        self.d1 = down(base_channals, base_channals * 2)
        self.d2 = down(base_channals * 2, base_channals * 4)
        self.d3 = down(base_channals * 4, base_channals * 8)
        factor = 2 if bilinear else 1
        self.d4 = down(base_channals * 8, base_channals * 16 // factor)
        #设置瓶颈
        bottleneck_channals=base_channals*16//factor
        self.attention=ChannalAttention(bottleneck_channals)if use_attention else nn.Identity

        self.u1=up(base_channals*16,base_channals*8//factor,bilinear)
        self.u2=up(base_channals*8,base_channals*4//factor,bilinear)
        self.u3=up(base_channals*4,base_channals*2//factor,bilinear)
        self.u1=up(base_channals*2,base_channals,bilinear)
        self.out=nn.Conv2d(base_channals,out_channals,1)

    def forward(self,x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x5 = self.attnention(x5)#瓶颈
        # 解码器 + 跳跃连接
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)