# -*- coding: utf-8 -*-

import os
import cv2
import random
import torch
import torch.nn as nn
import torchvision as tv
from glob import glob
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, aligned=False):
        self.transform = transforms
        self.aligned = aligned

        self.files_A = sorted(glob(os.path.join(root, "no_label_negative/*.*")))
        self.files_B = sorted(glob(os.path.join(root, "no_label_positive/*.*")))

    def __getitem__(self, index):
        image_B = cv2.imread(self.files_B[index % len(self.files_B)])

        if self.aligned:
            image_A = cv2.imread(self.files_A[index % len(self.files_A)])
        else:
            image_A = cv2.imread(self.files_A[random.randint(0, len(self.files_A)-1)])

        item_A = self.transform(image_A[:, :, 1])
        item_B = self.transform(image_B[:, :, 1])
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def init_weights(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv2") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()
        self.register_buffer('real_label', torch.ones(1))
        self.register_buffer('fake_label', torch.zeros(1))

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
        else:
            target_tensor = self.fake_label.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss


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


class DeconvBlock(nn.Module):
    """Deconvolution Block"""

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
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride,  padding=padding,
                               padding_mode=padding_mode, bias=bias),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        )


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
