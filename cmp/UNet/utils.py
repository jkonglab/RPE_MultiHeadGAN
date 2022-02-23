# -*- coding: utf-8 -*-

import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision as tv
from glob import glob
from torch.utils.data import Dataset


transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.RandomVerticalFlip(),
    tv.transforms.RandomHorizontalFlip()
])


def pairedTransform(x, y):
    img = np.stack((x, y), axis=2)
    tensor = transforms(img)
    tx, ty = torch.split(tensor, 1)
    tx = tv.transforms.Normalize(0.5, 0.5)(tx)
    return tx, ty


class ImageDataset(Dataset):
    def __init__(self, root, dir_A, dir_B):
        self.files_A = sorted(glob(os.path.join(root, dir_A, "*.*")))
        self.files_B = sorted(glob(os.path.join(root, dir_B, "*.*")))

    def __getitem__(self, index):
        image_B = cv2.imread(self.files_B[index % len(self.files_B)])
        image_A = cv2.imread(self.files_A[index % len(self.files_A)])

        item_A, item_B = pairedTransform(image_A[:, :, 1], image_B[:, :, 1])
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
