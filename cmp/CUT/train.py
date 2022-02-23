# -*- coding: utf-8 -*-

import itertools
import cv2
import time
import torch
import pandas as pd
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import *
from config import opt
from glob import glob


def train(**kwargs):
    """Training Network"""

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda:1" if opt.gpu else "cpu")
    setattr(opt, "device", device)

    # 1. Load data
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize(int(opt.img_sz*1.12)),
        tv.transforms.RandomCrop(opt.img_sz),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])

    dataloader = DataLoader(
        ImageDataset(opt.src_path, transforms=transforms),
        batch_size=opt.batch_sz,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # 2. Initialize network
    G_net = UNetGenerator(opt.n_gf, opt.n_mlp_dim, in_ch=1, out_ch=1, device=device)
    D_net = Discriminator(opt.n_df, in_ch=1, device=device)

    # 3. Define Optimizing Strategy
    optimizer_G = torch.optim.Adam(
        G_net.parameters(),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr
    )
    optimizer_D = torch.optim.Adam(
        D_net.parameters(),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr
    )
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_G,
        T_0=10,
        T_mult=2
    )
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_D,
        T_0=10,
        T_mult=2
    )
    criterion_idt = nn.L1Loss().to(device)
    criterion_GAN = GANLoss().to(device)
    criterion_NCE = PatchNCELoss(opt).to(device)

    # 4. Initialize Variables
    # threshold = torch.ones(1).to(device) * 0.5
    # true_labels = torch.ones(opt.batch_sz).to(device)
    # fake_labels = torch.zeros(opt.batch_sz).to(device)

    # 5. Create Validation set
    val_pos_path = os.path.join(opt.val_path, "positive/*.*")
    tensors = []
    for file_name in glob(val_pos_path):
        tmp_img = cv2.imread(file_name)
        tensors.append(transforms(tmp_img[:, :, 1]))
    val_real_pos = torch.stack(tensors).to(device)
    val_neg_path = os.path.join(opt.val_path, "negative/*.*")
    tensors = []
    for file_name in glob(val_neg_path):
        tmp_img = cv2.imread(file_name)
        tensors.append(transforms(tmp_img[:, :, 1]))
    val_real_neg = torch.stack(tensors).to(device)

    # 6. Train Networks
    log_path = opt.log_path + "CUT-UNet" + \
               time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    writer = SummaryWriter(log_path)
    for epoch in range(opt.max_epoch):
        # Train
        for i, batch in enumerate(dataloader):
            G_net.train()
            D_net.train()

            real_pos = batch["no_label_positive"].to(device)
            real_neg = batch["no_label_negative"].to(device)

            # Forward
            idt_pos = G_net(real_pos, result_only=True)
            fake_pos, feature_real_neg = G_net(real_neg)
            feature_fake_pos = G_net(fake_pos, feature_only=True)

            # Train Generator
            optimizer_G.zero_grad()

            loss_idt = criterion_idt(real_pos, idt_pos)
            loss_gan = criterion_GAN(D_net(fake_pos), True)
            loss_nce = 0
            for fr, ff in zip(feature_real_neg, feature_fake_pos):
                loss_nce += criterion_NCE(fr, ff)
            loss_nce /= len(feature_real_neg)
            loss_G = loss_gan + loss_nce * opt.lambda_nce + loss_idt * opt.lambda_idt

            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            loss_D_real = criterion_GAN(D_net(real_pos), True)
            loss_D_fake = criterion_GAN(D_net(fake_pos.detach()), False)
            loss_D = (loss_D_real + loss_D_fake) / 2

            loss_D.backward()
            optimizer_D.step()

        scheduler_G.step()
        scheduler_D.step()

        # 7. Validate
        G_net.eval()
        D_net.eval()

        idt_pos = G_net(val_real_pos, result_only=True)
        fake_pos, feature_real_neg = G_net(val_real_neg)
        feature_fake_pos = G_net(fake_pos, feature_only=True)

        loss_idt = criterion_idt(val_real_pos, idt_pos)
        loss_gan = criterion_GAN(D_net(fake_pos), True)
        loss_nce = 0
        for fr, ff in zip(feature_real_neg, feature_fake_pos):
            loss_nce += criterion_NCE(fr, ff)
        loss_nce /= len(feature_real_neg)
        loss_G = loss_gan + loss_nce * opt.lambda_nce + loss_idt * opt.lambda_idt

        writer.add_scalar("loss_G", loss_G.item(), epoch)
        writer.add_scalar("loss_gan_gray", loss_gan.item(), epoch)
        writer.add_scalar("loss_idt_gray", loss_idt.item(), epoch)
        writer.add_scalar("loss_nce", loss_nce.item(), epoch)

        loss_D_real = criterion_GAN(D_net(val_real_pos), True)
        loss_D_fake = criterion_GAN(D_net(fake_pos), False)
        loss_D = (loss_D_real + loss_D_fake) / 2

        writer.add_scalar("loss_D_gray", loss_D.item(), epoch)

        cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
        print("epoch:", epoch, cur_time)

        # 8. Save models
        if (epoch + 1) % opt.save_freq == 0:
            torch.save(G_net.get_param(), "%s/G_net_%s.pth" % (opt.model_path, epoch+1))
            img_rec = tv.transforms.Normalize(-1, 2)
            writer.add_image("idt_pos/%d" % (epoch+1), tv.utils.make_grid(img_rec(idt_pos), 4), epoch+1)
            writer.add_image("fake_pos/%d" % (epoch+1), tv.utils.make_grid(img_rec(fake_pos), 4), epoch+1)

    writer.close()



if __name__ == "__main__":
    train(src_path="train", max_epoch=400, gpu=True, save_freq=50, batch_sz=64, log_path="logs")
