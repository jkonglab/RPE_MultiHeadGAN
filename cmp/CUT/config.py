# -*- coding: utf-8 -*-


class Config(object):
    # path
    src_path = "data"
    dst_path = "results"
    val_path = "validation"
    model_path = "checkpoints"

    # training parameters
    gpu = False
    img_sz = 64
    batch_sz = 64
    n_gf = 64
    n_df = 64
    n_mlp_dim = 256
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    max_epoch = 20
    nce_t = 0.07
    lambda_nce = 1
    lambda_idt = 5
    save_freq = 10

    # testing parameters


opt = Config()
