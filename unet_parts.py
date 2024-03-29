import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import bilinear


#两次卷积
class DoubleConv(nn.Module):
    def __init__(self,in_channels, outchannels):
        super().__init__()
        self.double_conv = nn.Sequential(      #一个容器，可以按顺序包含多个神经网络层或模块，并构成一个整体的神经网络序列
            nn.Conv2d(in_channels,outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

#下采样模块
class Down(nn.Module):
    def __init__(self, in_channels, outchannels):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, outchannels)
        )

    def forward(self, x):
        return self.downsampling(x)

#上采样+特征融合
class Up(nn.Module):
    def __init__(self, in_channels, outchannels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, outchannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#输出
class outConv(nn.Module):
    def __init__(self, in_channels, outchannels):
        super(outConv, self).__init__()
        self.outconv = nn.Conv2d(in_channels, outchannels, kernel_size=1, padding=0)
    def forward(self, x):
        x = self.outconv(x)
        return x