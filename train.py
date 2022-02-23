import itertools
import cv2
import time
import torch
import pandas as pd
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from utils import *
from model import *
from config import opt
from glob import glob


def train(**kwargs):
    """Training Network"""

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda:0" if opt.gpu else "cpu")
    setattr(opt, "device", device)

    # 1. Load data
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.RandomCrop(opt.img_sz),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor()
    ])

    dataloader = DataLoader(
        ImageDataset(opt.train_path, transforms=transforms),
        batch_size=opt.batch_sz,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # 2. Initialize network
    G_net = UNetGenerator(opt.n_gf, opt.n_mlp_dim, in_ch=1, out_ch=1, multihead=True, device=device)
    D_net_gray = Discriminator(opt.n_df, in_ch=1, device=device)
    D_net_bw = Discriminator(opt.n_df, in_ch=1, device=device)


    # 3. Define Optimizing Strategy
    optimizer_G = torch.optim.Adam(
        G_net.parameters(),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr_g
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(D_net_gray.parameters(), D_net_bw.parameters()),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr_d
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
    criterion_shape = ShapeLoss().to(device)

    # 4. Initialize Variables
    fake_gray_pool = ImagePool(800)
    fake_bw_pool = ImagePool(800)


    # 5. Train Networks
    def mix_loss(loss1, loss2, epoch, start, end, max_ratio, min_ratio):
        if epoch < start:
            ratio = max_ratio
        elif epoch > end:
            ratio = min_ratio
        else:
            ratio = ((epoch - start) * max_ratio +
                    (end - epoch) * min_ratio) / (end - start)
        return loss1 * ratio + loss2 * (1 - ratio)

    for epoch in range(opt.max_epoch):
        # Train
        for i, batch in enumerate(dataloader):
            G_net.train()
            D_net_gray.train()
            D_net_bw.train()

            real_pos = batch["no_label_positive"].to(device)
            real_neg = batch["no_label_negative"].to(device)
            input = batch["label_input"].to(device)
            target = batch["label_target"].to(device)

            # Forward
            idt_pos = G_net(real_pos, result_only=True)
            fake_pos, feature_real_neg = G_net(real_neg)
            feature_fake_pos = G_net(fake_pos, feature_only=True)
            pred_input = G_net(input, bw_output=True)
            pred_real_neg = G_net(real_neg, bw_output=True)

            # Train Generator
            optimizer_G.zero_grad()

            loss_gan_gray = criterion_GAN(D_net_gray(fake_pos), True)
            loss_idt_gray = criterion_idt(real_pos, idt_pos)
            loss_nce = 0
            for fr, ff in zip(feature_real_neg, feature_fake_pos):
                loss_nce += criterion_NCE(fr, ff)
            loss_nce /= len(feature_real_neg)
            loss_G_gray = loss_gan_gray + loss_nce * opt.lambda_nce + \
                     loss_idt_gray * opt.lambda_idt

            loss_gan_bw = criterion_GAN(D_net_bw(pred_real_neg), True)
            loss_idt_bw = criterion_idt(pred_input, target)
            loss_shape = criterion_shape(pred_input, target)
            loss_G_bw = loss_gan_bw + (loss_idt_bw + loss_shape) * opt.lambda_idt

            loss_G = mix_loss(loss_G_gray, loss_G_bw, epoch, 40, 70, 1, 0.7)

            loss_G.backward()
            optimizer_G.step()

            # Update fake pool
            fake_pos = fake_gray_pool.query(fake_pos)
            pred_real_neg = fake_bw_pool.query(pred_real_neg)

            # Train Discriminator
            optimizer_D.zero_grad()

            loss_D_gray_real = criterion_GAN(D_net_gray(real_pos), True)
            loss_D_gray_fake = criterion_GAN(D_net_gray(fake_pos.detach()), False)
            loss_D_gray = loss_D_gray_real + loss_D_gray_fake
            loss_D_bw_real = criterion_GAN(D_net_bw(target), True)
            loss_D_bw_fake = criterion_GAN(D_net_bw(pred_real_neg.detach()), False)
            loss_D_bw = loss_D_bw_real + loss_D_bw_fake
            loss_D = mix_loss(loss_D_gray, loss_D_bw, epoch, 40, 70, 1, 0.7)

            loss_D.backward()
            optimizer_D.step()

        scheduler_G.step()
        scheduler_D.step()

        cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
        print("epoch:", epoch, cur_time)

        # Save models
        if (epoch + 1) % opt.save_freq == 0:
            torch.save(G_net.get_param(), "%s/G_net_%s.pth" % (opt.model_path, epoch+1))



if __name__ == "__main__":
    train(train_path="train", max_epoch=200, gpu=True, save_freq=50, batch_sz=32)
