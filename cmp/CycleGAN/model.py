import torch.nn as nn
import torch
from utils import *


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


# class UNetGenerator(nn.Module):
#     """Use U-Net as Generator"""
#
#     def __init__(self, n_f, in_ch=3, out_ch=3):
#         super().__init__()
#
#         filters = [n_f, n_f*2, n_f*4, n_f*8, n_f*16]
#
#         # self.norm = nn.InstanceNorm2d(in_ch)
#
#         self.conv1 = DoubleConvBlock(in_ch, filters[0], norm="batch")
#         self.conv2 = DoubleConvBlock(filters[0], filters[1], norm="batch")
#         self.conv3 = DoubleConvBlock(filters[1], filters[2], norm="batch")
#         self.conv4 = DoubleConvBlock(filters[2], filters[3], norm="batch")
#         self.conv5 = DoubleConvBlock(filters[3], filters[4], norm="batch")
#
#         self.mp1 = nn.MaxPool2d(2)
#         self.mp2 = nn.MaxPool2d(2)
#         self.mp3 = nn.MaxPool2d(2)
#         self.mp4 = nn.MaxPool2d(2)
#
#         self.us1 = UpConvBlock(filters[4], filters[3], norm="batch")
#         self.us2 = UpConvBlock(filters[3], filters[2], norm="batch")
#         self.us3 = UpConvBlock(filters[2], filters[1], norm="batch")
#         self.us4 = UpConvBlock(filters[1], filters[0], norm="batch")
#
#         self.conv6 = DoubleConvBlock(filters[4], filters[3], norm="batch")
#         self.conv7 = DoubleConvBlock(filters[3], filters[2], norm="batch")
#         self.conv8 = DoubleConvBlock(filters[2], filters[1], norm="batch")
#         self.conv9 = DoubleConvBlock(filters[1], filters[0], norm="batch")
#
#         self.output = nn.Conv2d(filters[0], out_ch, 1)
#
#     def forward(self, x):
#         # Down sampling
#         # n = self.norm(x)
#         e1 = self.conv1(x)
#         m1 = self.mp1(e1)
#         e2 = self.conv2(m1)
#         m2 = self.mp2(e2)
#         e3 = self.conv3(m2)
#         m3 = self.mp3(e3)
#         e4 = self.conv4(m3)
#         m4 = self.mp4(e4)
#         e5 = self.conv5(m4)
#
#         # Up sampling
#         u1 = self.us1(e5)
#         d1 = self.conv6(torch.cat([e4, u1], dim=1))
#         u2 = self.us2(d1)
#         d2 = self.conv7(torch.cat([e3, u2], dim=1))
#         u3 = self.us3(d2)
#         d3 = self.conv8(torch.cat([e2, u3], dim=1))
#         u4 = self.us4(d3)
#         d4 = self.conv9(torch.cat([e1, u4], dim=1))
#
#         return torch.tanh(self.output(d4))

class UNetGenerator(nn.Module):
    """Use U-Net as Generator"""

    def __init__(self, n_f, in_ch=3, out_ch=3):
        super().__init__()

        filters = [n_f, n_f*2, n_f*4, n_f*8]

        # self.norm = nn.InstanceNorm2d(in_ch)

        self.conv1 = DoubleConvBlock(in_ch, filters[0], norm="batch")
        self.conv2 = DoubleConvBlock(filters[0], filters[1], norm="batch")
        self.conv3 = DoubleConvBlock(filters[1], filters[2], norm="batch")
        self.conv4 = DoubleConvBlock(filters[2], filters[3], norm="batch")

        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
        self.mp3 = nn.MaxPool2d(2)

        self.us1 = UpConvBlock(filters[3], filters[2], norm="batch")
        self.us2 = UpConvBlock(filters[2], filters[1], norm="batch")
        self.us3 = UpConvBlock(filters[1], filters[0], norm="batch")

        self.conv5 = DoubleConvBlock(filters[3], filters[2], norm="batch")
        self.conv6 = DoubleConvBlock(filters[2], filters[1], norm="batch")
        self.conv7 = DoubleConvBlock(filters[1], filters[0], norm="batch")

        self.output = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        # Down sampling
        # n = self.norm(x)
        e1 = self.conv1(x)
        m1 = self.mp1(e1)
        e2 = self.conv2(m1)
        m2 = self.mp2(e2)
        e3 = self.conv3(m2)
        m3 = self.mp3(e3)
        e4 = self.conv4(m3)

        # Up sampling
        u1 = self.us1(e4)
        d1 = self.conv5(torch.cat([e3, u1], dim=1))
        u2 = self.us2(d1)
        d2 = self.conv6(torch.cat([e2, u2], dim=1))
        u3 = self.us3(d2)
        d3 = self.conv7(torch.cat([e1, u3], dim=1))

        return torch.tanh(self.output(d3))


class Discriminator(nn.Module):
    """Down Sampling Discriminator"""

    def __init__(self, n_f, in_ch=3):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch, n_f, 4, stride=2, padding=1, padding_mode="reflect"),    # 96 -> 48
            ConvBlock(n_f, n_f*2, 4, stride=2, padding=1, padding_mode="reflect"),    # 48 -> 24
            ConvBlock(n_f*2, n_f*4, 4, stride=2, padding=1, padding_mode="reflect"),  # 24 -> 12
            ConvBlock(n_f*4, n_f*8, 4, stride=2, padding=1, padding_mode="reflect"),  # 12 -> 6
            nn.Conv2d(n_f*8, 1, 1)                                                    # 6 -> 1
        )

    def forward(self, x):
        return self.net(x)
