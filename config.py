# -*- coding: utf-8 -*-


class Config(object):
    # path
    train_path = "data"         # path to load training data
    model_path = "checkpoints"  # path to save (intermediate) models

    # training parameters
    gpu = False     # whether to train with GPU
    img_sz = 64     # image size
    batch_sz = 64   # batch size
    n_gf = 64       # number of filters in the generator at the highest level
    n_df = 64       # number of filters in the discriminator at the highest level
    n_mlp_dim = 256 # number of units in each MLP layer
    lr_g = 2e-4     # generator learning rate
    lr_d = 4e-4     # discriminator learning rate
    beta1 = 0.5     # parameter 1 for Adam optimizer
    beta2 = 0.999   # parameter 2 for Adam optimizer
    max_epoch = 20  # max epoch number
    nce_t = 0.07    # scaling factor in NCE loss
    lambda_nce = 1  # weight factor for NCE loss
    lambda_idt = 1  # weight factor for Identity loss
    save_freq = 10  # saving frequency


opt = Config()
