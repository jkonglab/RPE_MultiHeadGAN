import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from utils import *
from model import *
from config import opt


@torch.no_grad()
def predict(model_path, test_path, dst_path, transforms):

    device = torch.device("cpu")

    G_net = UNetGenerator(opt.n_gf, opt.n_mlp_dim, in_ch=1, out_ch=1, multihead=True, device=device).eval()
    G_net.load_state_dict(torch.load(model_path, map_location=device), strict=False)


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
        Y = G_net(X, bw_output=True)
        output = Y.squeeze().cpu().numpy()
        output = output[0:height, 0:width]
        output = ((output>0)*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path, name), output)



if __name__ == "__main__":
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])

    model_path = "checkpoints/G_net_100.pth"
    test_path = "test"
    dst_path = "results"

    predict(model_path, test_path, dst_path, transforms)
