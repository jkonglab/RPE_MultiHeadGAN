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
from config import opt
from glob import glob


def train(**kwargs):
    """Training Network"""

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda:0" if opt.gpu else "cpu")

    # 1. Load data
    dataloader = DataLoader(
        ImageDataset(opt.src_path, "input", "target"),
        batch_size=opt.batch_sz,
        shuffle=True,
        drop_last=True,
        num_workers=1
    )

    # 2. Initialize network
    model = tv.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False, num_classes=1)
    # model.apply(init_weights)
    model.to(device)

    # 3. Define Optimizing Strategy
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    criterion = nn.BCEWithLogitsLoss().to(device)

    log_path = opt.log_path + "DeepLab_" + \
               time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    writer = SummaryWriter(log_path)

    # 6. Train Networks
    for epoch in range(opt.max_epoch):
        # Train
        for i, batch in enumerate(dataloader):
            model.train()
            image = batch["A"].to(device)
            label = batch["B"].to(device)

            # Forward
            prediction = model(image)['out']

            # Backward
            optimizer.zero_grad()
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        scheduler.step()
        writer.add_scalar("loss_BCE", loss.item(), epoch)
        cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
        print("epoch:", epoch, cur_time)

        # 8. Save models
        if (epoch + 1) % opt.save_freq == 0:
            torch.save(model.state_dict(), "%s/model_%s.pth" % (opt.model_path, epoch+1))


if __name__ == "__main__":
    train(src_path="train", model_path = "checkpoints", max_epoch=200, gpu=True, save_freq=50, batch_sz=4)
