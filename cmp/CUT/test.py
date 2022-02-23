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

    G_net = ResidualGenerator(opt.n_gf, in_ch=1, out_ch=1, device=device).eval()
    G_net.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    for path in glob(os.path.join(test_path, '*.tif')):
        name = os.path.basename(path)
        img = cv2.imread(path)
        X = transforms(img[:, :, 1]).to(device)
        X.unsqueeze_(0)
        Y = G_net(X, result_only=True)
        output = Y.squeeze().cpu().numpy()
        output = ((output+1)/2*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path, name), output)



if __name__ == "__main__":
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize(opt.img_sz),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])

    model_path = "checkpoints/G_net_400.pth"
    test_path = "test"
    dst_path = "results"

    predict(model_path, test_path, dst_path, transforms)
