import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from utils import *
from model import *
from config import opt
from torchstat import stat
from datetime import datetime, timedelta


@torch.no_grad()
def predict(model_path, test_path, dst_path, transforms):

    device = torch.device("cpu")

    G_net = UNetGenerator(opt.n_gf, in_ch=1, out_ch=1).eval()
    G_net.to(device)
    G_net.load_state_dict(torch.load(model_path, map_location=device))

    criterion_GAN = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    acc = 0
    num = 0
    for path in glob(os.path.join(test_path, '*.tif')):
        name = os.path.basename(path)
        img = cv2.imread(path)
        img = img[:, :, 1]
        height, width = img.shape
        height_pad = (8 - height % 8) % 8
        width_pad = (8 - width % 8) % 8
        img = cv2.copyMakeBorder(img, 0, height_pad, 0, width_pad, cv2.BORDER_CONSTANT, value=0)
        X = transforms(img).to(device)
        X.unsqueeze_(0)
        s = datetime.now()
        Y = G_net(X)
        e = datetime.now()
        acc += (e-s) / timedelta(seconds=1)
        num += 1
        output = Y.squeeze().cpu().numpy()
        output = ((output[:height, :width]+1)/2*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path, name), output)
    print("average time:", acc/num)



if __name__ == "__main__":
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        # tv.transforms.Resize(opt.img_sz),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])

    model_path = "checkpoints/G_net_400.pth"
    test_path = "test"
    dst_path = "results"

    predict(model_path, test_path, dst_path, transforms)
