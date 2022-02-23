import torch.nn as nn
import torch
from utils import *


class ConvBlock(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
             padding_mode="zeros", bias=False, norm='batch'):
        super().__init__()

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      padding_mode=padding_mode, bias=bias),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, ch, norm='batch'):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(ch, ch, 3, stride=1, padding=1, padding_mode="reflect", norm='batch'),
            ConvBlock(ch, ch, 3, stride=1, padding=1, padding_mode="reflect", norm='batch'),
        )

    def forward(self, x):
        return x + self.net(x)


class DoubleConvBlock(nn.Module):
    """Two Convolution Layers"""

    def __init__(self, in_ch, out_ch, norm="batch"):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect", norm=norm),
            ConvBlock(out_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect", norm=norm)
        )

    def forward(self, x):
        return self.net(x)


class UpConvBlock(nn.Module):
    """Up-Sampling Layer plus Convolution Layer"""

    def __init__(self, in_ch, out_ch, norm="batch"):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect", norm=norm)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):

    def __init__(self, n_f, in_ch=3, out_ch=3):
        super().__init__()

        filters = [n_f, n_f*2, n_f*4, n_f*8, n_f*16]

        self.conv1 = DoubleConvBlock(in_ch, filters[0], norm="instance")
        self.conv2 = DoubleConvBlock(filters[0], filters[1], norm="instance")
        self.conv3 = DoubleConvBlock(filters[1], filters[2], norm="instance")
        self.conv4 = DoubleConvBlock(filters[2], filters[3], norm="instance")
        self.conv5 = DoubleConvBlock(filters[3], filters[4], norm="instance")

        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
        self.mp3 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(2)

        self.us1 = UpConvBlock(filters[4], filters[3], norm="instance")
        self.us2 = UpConvBlock(filters[3], filters[2], norm="instance")
        self.us3 = UpConvBlock(filters[2], filters[1], norm="instance")
        self.us4 = UpConvBlock(filters[1], filters[0], norm="instance")

        self.conv6 = DoubleConvBlock(filters[4], filters[3], norm="instance")
        self.conv7 = DoubleConvBlock(filters[3], filters[2], norm="instance")
        self.conv8 = DoubleConvBlock(filters[2], filters[1], norm="instance")
        self.conv9 = DoubleConvBlock(filters[1], filters[0], norm="instance")

        self.output = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        # Down sampling
        e1 = self.conv1(x)
        m1 = self.mp1(e1)
        e2 = self.conv2(m1)
        m2 = self.mp2(e2)
        e3 = self.conv3(m2)
        m3 = self.mp3(e3)
        e4 = self.conv4(m3)
        m4 = self.mp4(e4)
        e5 = self.conv5(m4)

        # Up sampling
        u1 = self.us1(e5)
        d1 = self.conv6(torch.cat([e4, u1], dim=1))
        u2 = self.us2(d1)
        d2 = self.conv7(torch.cat([e3, u2], dim=1))
        u3 = self.us3(d2)
        d3 = self.conv8(torch.cat([e2, u3], dim=1))
        u4 = self.us4(d3)
        d4 = self.conv9(torch.cat([e1, u4], dim=1))

        return self.output(d4)
