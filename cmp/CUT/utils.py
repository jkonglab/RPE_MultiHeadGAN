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


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.files_A = sorted(glob(os.path.join(root, "no_label_positive/*.*")))
        self.files_B = sorted(glob(os.path.join(root, "no_label_negative/*.*")))
        self.files_C = sorted(glob(os.path.join(root, "label_input/*.*")))
        self.files_D = sorted(glob(os.path.join(root, "label_target/*.*")))

    def __getitem__(self, index):
        image_A = cv2.imread(self.files_A[index % len(self.files_A)])
        image_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B)-1)])
        rand_idx = random.randint(0, len(self.files_C)-1)
        image_C = cv2.imread(self.files_C[rand_idx])
        image_D = cv2.imread(self.files_D[rand_idx])

        def pairedTransform(x, y, transform):
            img = np.stack((x, y), axis=2)
            tensor = transform(img)
            tx, ty = torch.split(tensor, 1)
            tx = tv.transforms.Normalize(0.5, 0.5)(tx)
            ty = tv.transforms.Normalize(0.5, 0.5)(ty)
            return tx, ty

        def Normalize(input):
            return tv.transforms.Normalize(0.5, 0.5)(input)

        item_A = Normalize(self.transform(image_A[:, :, 1]))
        item_B = Normalize(self.transform(image_B[:, :, 1]))
        item_C, item_D = pairedTransform(image_C[:, :, 1], image_D[:, :, 1], self.transform)
        return {"no_label_positive": item_A, "no_label_negative": item_B,
                "label_input": item_C, "label_target": item_D}

    def __len__(self):
        return len(self.files_A)

# class ImageDataset(Dataset):
#     def __init__(self, root, transforms=None, aligned=False):
#         self.transform = transforms
#         self.aligned = aligned
#
#         self.files_A = sorted(glob(os.path.join(root, "positive/*.*")))
#         self.files_B = sorted(glob(os.path.join(root, "negative/*.*")))
#
#     def __getitem__(self, index):
#         image_B = cv2.imread(self.files_B[index % len(self.files_B)])
#
#         if self.aligned:
#             image_A = cv2.imread(self.files_A[index % len(self.files_A)])
#         else:
#             image_A = cv2.imread(self.files_A[random.randint(0, len(self.files_A)-1)])
#
#         item_A = self.transform(image_A[:, :, 1])
#         item_B = self.transform(image_B[:, :, 1])
#         return {"A": item_A, "B": item_B}
#
#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))


class ImagePool():
    def __init__(self, size):
        self.size = size
        if self.size > 0:
            self.num = 0
            self.images = []

    def query(self, images):
        if self.size == 0:
            return images
        return_images = []
        for image in images:
            image = image.data
            if self.num < self.size:
                self.num += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    rand_id = random.randint(0, self.size-1)
                    out_image = self.images[rand_id].clone()
                    self.images[rand_id] = image
                    return_images.append(out_image)
                else:
                    return_images.append(image)
        return_images = torch.stack(return_images, 0)
        return return_images


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0/self.power)
        return x.div(norm + 1e-7)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()
        self.register_buffer("real_label", torch.ones(1))
        self.register_buffer("fake_label", torch.zeros(1))

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
        else:
            target_tensor = self.fake_label.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss


class ShapeLoss(nn.Module):
    def __init__(self, factor=1):
        super().__init__()
        self.register_buffer("ref_label", torch.ones(1))
        self.factor = factor

    def __call__(self, input, target):
        ref_tensor = self.ref_label.expand_as(input)
        input = (input + 1) / 2
        target = (target + 1) / 2
        loss = nn.functional.binary_cross_entropy(input, target, reduction="none")
        weight = target * self.factor + ref_tensor * (1 - self.factor)
        return torch.mean(weight*loss)


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        #     batch_dim_for_bmm = self.opt.batch_sz

        # reshape features to batch size
        feat_q = feat_q.view(1, -1, dim)
        feat_k = feat_k.view(1, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_t
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss


class ConvBlock(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 padding_mode="zeros", bias=False, norm="batch", activation="relu"):
        super().__init__()

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "hardswish":
            activation_layer = nn.Hardswish(inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      padding_mode=padding_mode, bias=bias),
            norm_layer(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    """Deconvolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=False, norm="batch", activation="relu"):
        super().__init__()

        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "hardswish":
            activation_layer = nn.Hardswish(inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, bias=bias),
            norm_layer(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, ch, norm="batch"):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(ch, ch, 3, stride=1, padding=1, padding_mode="reflect", norm='batch'),
            ConvBlock(ch, ch, 3, stride=1, padding=1, padding_mode="reflect", norm='batch'),
        )

    def forward(self, x):
        return x + self.net(x)


class ResBlockV2(nn.Module):
    """Residual Block v2"""
    def __init__(self, ch, norm="batch", activation="relu"):
        super().__init__()

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            norm_layer(ch),
            activation_layer,
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect"),
            norm_layer(ch),
            activation_layer,
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")
        )

    def forward(self, x):
        return x + self.net(x)


class SpatialAttenBlock(nn.Module):
    """Spatial Attention Block"""

    def __init__(self, kernel_size):
        super().__init__()

        self.attention = ConvBlock(2, 1, kernel_size, activation=None,
                                   padding=(kernel_size-1) // 2,
                                   padding_mode="reflect")

    def forward(self, x):
        x_compress = torch.stack((torch.amax(x, 1), torch.mean(x, 1)), dim=1)
        x_out = self.attention(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class ResidualSpatialAttenBlock(nn.Module):
    """Residual Block with Spatial Attention"""

    def __init__(self, in_ch, out_ch, stride=1, norm="batch", activation="relu"):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, stride=stride, padding=1,
                      padding_mode="reflect", norm=norm, activation=activation),
            ConvBlock(out_ch, out_ch, 3, padding=1, padding_mode="reflect",
                      norm=norm, activation=activation)
        )
        self.attention = SpatialAttenBlock(3)
        self.shortcut = nn.Identity()
        if stride > 1 or out_ch != in_ch:
            self.shortcut = ConvBlock(in_ch, out_ch, 1, stride=stride, activation=None)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.attention(self.net(x)) + self.shortcut(x))


class DoubleConvBlock(nn.Module):
    """Two Convolution Layers"""

    def __init__(self, in_ch, out_ch, norm="batch", activation="relu"):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, norm=norm,
                      padding_mode="reflect", activation=activation),
            ConvBlock(out_ch, out_ch, 3, stride=1, padding=1, norm=norm,
                      padding_mode="reflect", activation=activation)
        )

    def forward(self, x):
        return self.net(x)


class UpConvBlock(nn.Module):
    """Up-Sampling Layer plus Convolution Layer"""

    def __init__(self, in_ch, out_ch, norm="batch", activation="relu"):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_ch, out_ch, 3, stride=1, padding=1, norm=norm,
                      padding_mode="reflect", activation=activation)
        )

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv2") != -1 or class_name.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif class_name.find("Norm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
