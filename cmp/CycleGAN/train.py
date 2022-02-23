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

    # 1. Load data
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.RandomCrop(opt.img_sz),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])

    dataloader = DataLoader(
        ImageDataset(opt.src_path, transforms=transforms, aligned=False),
        batch_size=opt.batch_sz,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # 2. Initialize network
    G_AB = UNetGenerator(opt.n_gf, in_ch=1, out_ch=1)
    G_BA = UNetGenerator(opt.n_gf, in_ch=1, out_ch=1)
    D_A = Discriminator(opt.n_df, in_ch=1)
    D_B = Discriminator(opt.n_df, in_ch=1)

    G_AB.apply(init_weights)
    G_BA.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)

    # 3. Define Optimizing Strategy
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr
    )
    scheduler_G = torch.optim.lr_scheduler.StepLR(
        optimizer_G,
        step_size=100,
        gamma=0.5
    )
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D,
        step_size=100,
        gamma=0.5
    )
    criterion_GAN = GANLoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    # 4. Initialize Variables
    # true_labels = torch.ones(opt.batch_sz).to(device)
    # fake_labels = torch.zeros(opt.batch_sz).to(device)

    # 5. Create Validation set
    val_pos_path = os.path.join(opt.val_path, "positive/*.*")
    tensors = []
    for file_name in glob(val_pos_path):
        tmp_img = cv2.imread(file_name)
        tensors.append(transforms(tmp_img[:, :, 1]))
    val_A = torch.stack(tensors).to(device)
    val_neg_path = os.path.join(opt.val_path, "negative/*.*")
    tensors = []
    for file_name in glob(val_neg_path):
        tmp_img = cv2.imread(file_name)
        tensors.append(transforms(tmp_img[:, :, 1]))
    val_B = torch.stack(tensors).to(device)

    log_path = opt.log_path + "cycleGAN-UNet_" + \
               time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    writer = SummaryWriter(log_path)

    # 6. Train Networks
    for epoch in range(opt.max_epoch):
        # Train
        for i, batch in enumerate(dataloader):
            G_AB.train()
            G_BA.train()
            D_A.train()
            D_B.train()

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Forward
            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)
            idt_B = G_AB(real_B)

            # Train Generator
            optimizer_G.zero_grad()

            loss_identity_B = criterion_identity(real_B, idt_B)
            loss_identity = loss_identity_B

            loss_cycle_A = criterion_cycle(real_A, rec_A)
            loss_cycle_B = criterion_cycle(real_B, rec_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B * 3) / 4

            loss_GAN_A = criterion_GAN(torch.squeeze(D_A(fake_A)), True)
            loss_GAN_B = criterion_GAN(torch.squeeze(D_B(fake_B)), True)
            loss_GAN = (loss_GAN_A + loss_GAN_B * 3) / 4

            loss_G = loss_GAN + loss_cycle * opt.lambda_cycle + \
                loss_identity * opt.lambda_identity

            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            loss_D_A_real = criterion_GAN(torch.squeeze(D_A(real_A)), True)
            loss_D_A_fake = criterion_GAN(torch.squeeze(D_A(fake_A.detach())), False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2
            loss_D_A.backward()

            loss_D_B_real = criterion_GAN(torch.squeeze(D_B(real_B)), True)
            loss_D_B_fake = criterion_GAN(torch.squeeze(D_B(fake_B.detach())), False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2
            loss_D_B.backward()

            optimizer_D.step()

        scheduler_G.step()
        scheduler_D.step()

        # 7. Validate
        G_AB.eval()
        G_BA.eval()
        D_A.eval()
        D_B.eval()

        fake_B = G_AB(val_A)
        rec_A = G_BA(fake_B)
        fake_A = G_BA(val_B)
        rec_B = G_AB(fake_A)

        loss_identity_A = criterion_identity(val_A, fake_B)
        loss_identity_B = criterion_identity(val_B, fake_A)
        loss_identity = loss_identity_A + loss_identity_B

        loss_cycle_A = criterion_cycle(val_A, rec_A)
        loss_cycle_B = criterion_cycle(val_B, rec_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        loss_GAN_A = criterion_GAN(torch.squeeze(D_A(fake_A)), True)
        loss_GAN_B = criterion_GAN(torch.squeeze(D_B(fake_B)), True)
        loss_GAN = loss_GAN_A + loss_GAN_B

        loss_D_A_real = criterion_GAN(torch.squeeze(D_A(val_A)), True)
        loss_D_A_fake = criterion_GAN(torch.squeeze(D_A(fake_A.detach())), False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2

        loss_D_B_real = criterion_GAN(torch.squeeze(D_B(val_B)), True)
        loss_D_B_fake = criterion_GAN(torch.squeeze(D_B(fake_B.detach())), False)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2

        writer.add_scalar("loss_identity", loss_identity.item(), epoch)
        writer.add_scalar("loss_cycle", loss_cycle.item(), epoch)
        writer.add_scalar("loss_GAN", loss_GAN.item(), epoch)
        writer.add_scalar("loss_D_A", loss_D_A.item(), epoch)
        writer.add_scalar("loss_D_B", loss_D_B.item(), epoch)

        cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
        print("epoch:", epoch, cur_time)

        # 8. Save models
        if (epoch + 1) % opt.save_freq == 0:
            torch.save(G_AB.state_dict(), "%s/G_net_%s.pth" % (opt.model_path, epoch+1))
            norm = tv.transforms.Normalize(-1, 2)
            writer.add_image("fake_pos/%d" % (epoch+1), tv.utils.make_grid(norm(fake_B), 4), epoch+1)


if __name__ == "__main__":
    train(src_path="train", max_epoch=400, gpu=True, save_freq=50, batch_sz=64, log_path="logs")
