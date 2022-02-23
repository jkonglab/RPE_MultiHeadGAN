import torch.nn as nn
import torch
from utils import *


class ResidualGenerator(nn.Module):

    def __init__(self, n_f, n_mlp_dim=256, n_res=6, n_downsample=2,
                 in_ch=3, out_ch=3, device=torch.device("cpu")):
        super().__init__()

        model = [nn.Conv2d(in_ch, n_f, 7, padding=3, padding_mode="reflect"),
                 nn.BatchNorm2d(n_f),
                 nn.ReLU(inplace=True)]
        for i in range(n_downsample):
            factor = 2 ** i
            model += [nn.Conv2d(n_f*factor, n_f*factor*2, 3, stride=2,
                                padding=1, padding_mode="reflect"),
                      nn.BatchNorm2d(n_f*factor*2),
                      nn.ReLU(inplace=True)]

        factor = 2 ** n_downsample
        for i in range(n_res):
            model += [ResBlockV2(n_f*factor)]

        for i in range(n_downsample):
            factor = 2 ** (n_downsample - i - 1)
            model += [UpConvBlock(n_f*factor*2, n_f*factor)]
        model += [nn.Conv2d(n_f, out_ch, 7, padding=3, padding_mode="reflect"),
                  nn.Tanh()]

        self.generator = nn.Sequential(*model)
        self.generator.to(device)
        self.generator.apply(init_weights)

        self.mlp_init = False
        self.mlp_dim = n_mlp_dim
        self.n_downsample = n_downsample
        self.norm = Normalize()
        self.device = device
        self.n_f = n_f

    def create_mlp(self, x):
        init_dim = self.n_f * x.shape[2] * x.shape[3]
        for i in range(self.n_downsample+1):
            input_dim = int(init_dim / (2 ** i))
            mlp = nn.Sequential(
                nn.Linear(input_dim, self.mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.mlp_dim, self.mlp_dim)
            )
            mlp.to(self.device)
            mlp.apply(init_weights)
            setattr(self, "mlp_%d" % (i*3), mlp)
            self.mlp_init = True

    def forward(self, x, result_only=False, feature_only=False):
        if result_only:
            assert not feature_only, "There should be at least one output"
            return self.generator(x)
        else:
            if not self.mlp_init:
                self.create_mlp(x)
            tmp = x
            batch_sz = x.shape[0]
            features = []
            feature_layers = [i*3 for i in range(self.n_downsample+1)]
            for i, layer in enumerate(self.generator):
                tmp = layer(tmp)
                if i in feature_layers:
                    mlp = getattr(self, "mlp_%d" % i)
                    feature = mlp(tmp.view(batch_sz, -1))
                    features.append(self.norm(feature))
                if i == feature_layers[-1] and feature_only:
                    return features
            return tmp, features


class UNetGenerator(nn.Module):
    """Use U-Net as Generator"""

    def __init__(self, n_f, n_mlp_dim=256, in_ch=3, out_ch=3, multihead=False,
                 device=torch.device("cpu")):
        super().__init__()

        filters = [n_f, n_f*2, n_f*4, n_f*8]

        self.econv1 = DoubleConvBlock(in_ch, filters[0], norm="batch")
        self.econv2 = DoubleConvBlock(filters[0], filters[1], norm="batch")
        self.econv3 = DoubleConvBlock(filters[1], filters[2], norm="batch")
        self.econv4 = DoubleConvBlock(filters[2], filters[3], norm="batch")

        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
        self.mp3 = nn.MaxPool2d(2)

        self.us1 = UpConvBlock(filters[3], filters[2], norm="batch")
        self.us2 = UpConvBlock(filters[2], filters[1], norm="batch")
        self.us3 = UpConvBlock(filters[1], filters[0], norm="batch")

        self.dconv1 = DoubleConvBlock(filters[3], filters[2], norm="batch")
        self.dconv2 = DoubleConvBlock(filters[2], filters[1], norm="batch")
        self.dconv3 = DoubleConvBlock(filters[1], filters[0], norm="batch")
        self.output = nn.Conv2d(filters[0], out_ch, 1)

        self.multihead = False
        if multihead:
            self.multihead = True
            self.us1_bw = UpConvBlock(filters[3], filters[2], norm="batch")
            self.us2_bw = UpConvBlock(filters[2], filters[1], norm="batch")
            self.us3_bw = UpConvBlock(filters[1], filters[0], norm="batch")

            self.dconv1_bw = DoubleConvBlock(filters[3], filters[2], norm="batch")
            self.dconv2_bw = DoubleConvBlock(filters[2], filters[1], norm="batch")
            self.dconv3_bw = DoubleConvBlock(filters[1], filters[0], norm="batch")
            self.output_bw = nn.Conv2d(filters[0], out_ch, 1)

        self.to(device)
        self.apply(init_weights)

        self.mlp_init = False
        self.mlp_dim = n_mlp_dim
        self.n_downsample = 3
        self.norm = Normalize()
        self.n_f = n_f
        self.device = device

    def create_mlp(self, x):
        self.mlp_init = True
        init_dim = self.n_f * x.shape[2] * x.shape[3]
        for i in range(self.n_downsample+1):
            input_dim = int(init_dim / (2 ** i))
            mlp = nn.Sequential(
                nn.Linear(input_dim, self.mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.mlp_dim, self.mlp_dim)
            )
            mlp.to(self.device)
            mlp.apply(init_weights)
            setattr(self, "mlp_%d" % (i+1), mlp)

    def get_param(self, no_mlp=True):
        if no_mlp:
            simple_dict = {}
            whole_dict = self.state_dict()
            for k in whole_dict:
                if "mlp" not in k:
                    simple_dict[k] = whole_dict[k]
            return simple_dict
        else:
            return self.state_dict()

    def forward(self, x, bw_output=False, result_only=False, feature_only=False):
        # Down sampling
        e1 = self.econv1(x)
        m1 = self.mp1(e1)
        e2 = self.econv2(m1)
        m2 = self.mp2(e2)
        e3 = self.econv3(m2)
        m3 = self.mp3(e3)
        e4 = self.econv4(m3)

        # Up sampling
        if bw_output:
            assert self.multihead, "Black-white output is not initiated."
            u1_bw = self.us1_bw(e4)
            d1_bw = self.dconv1_bw(torch.cat([e3, u1_bw], dim=1))
            u2_bw = self.us2_bw(d1_bw)
            d2_bw = self.dconv2_bw(torch.cat([e2, u2_bw], dim=1))
            u3_bw = self.us3_bw(d2_bw)
            d3_bw = self.dconv3_bw(torch.cat([e1, u3_bw], dim=1))
            bw = torch.tanh(self.output_bw(d3_bw))
            return bw

        if not feature_only:
            u1 = self.us1(e4)
            d1 = self.dconv1(torch.cat([e3, u1], dim=1))
            u2 = self.us2(d1)
            d2 = self.dconv2(torch.cat([e2, u2], dim=1))
            u3 = self.us3(d2)
            d3 = self.dconv3(torch.cat([e1, u3], dim=1))
            gray = torch.tanh(self.output(d3))
            if result_only:
                return gray

        if not self.mlp_init:
            self.create_mlp(x)
        batch_sz = x.shape[0]
        f1 = self.norm(self.mlp_1(e1.view(batch_sz, -1)))
        f2 = self.norm(self.mlp_2(e2.view(batch_sz, -1)))
        f3 = self.norm(self.mlp_3(e3.view(batch_sz, -1)))
        f4 = self.norm(self.mlp_4(e4.view(batch_sz, -1)))
        features = [f1, f2, f3, f4]

        if feature_only:
            assert not result_only, "There should be at least one output"
            return features
        else:
            return gray, features


class Discriminator(nn.Module):
    """Down Sampling Discriminator"""

    def __init__(self, n_f, in_ch=3, device=torch.device("cpu")):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch, n_f, 4, stride=2, padding=1, padding_mode="reflect"),    # 96 -> 48
            ConvBlock(n_f, n_f*2, 4, stride=2, padding=1, padding_mode="reflect"),    # 48 -> 24
            ConvBlock(n_f*2, n_f*4, 4, stride=2, padding=1, padding_mode="reflect"),  # 24 -> 12
            ConvBlock(n_f*4, n_f*8, 4, stride=2, padding=1, padding_mode="reflect"),  # 12 -> 6
            nn.Conv2d(n_f*8, 1, 1)                                                    # 6 -> 1
        )
        self.net.to(device)
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)
