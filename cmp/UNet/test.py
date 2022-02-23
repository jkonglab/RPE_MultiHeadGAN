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

    model = UNet(opt.n_gf, in_ch=1, out_ch=1)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    for path in glob(os.path.join(test_path, '*.tif')):
        name = os.path.basename(path)
        img = cv2.imread(path)
        row, col, _ = img.shape
        pad_r = (16 - row % 16) % 16
        pad_c = (16 - col % 16) % 16
        X = transforms(img[:, :, 1]).to(device)
        pd = nn.ReflectionPad2d((0, pad_c, 0, pad_r))
        X.unsqueeze_(0)
        X = pd(X)
        Y = torch.sigmoid(model(X))
        output = Y.squeeze().cpu().numpy() > 0.5
        output = (output[:row, :col]*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path, name), output)


if __name__ == "__main__":
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])


    model_path = "checkpoints/model_100.pth"
    test_path = "test"
    dst_path = "results"

    predict(model_path, test_path, dst_path, transforms)
