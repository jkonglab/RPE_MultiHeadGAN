import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from utils import *
from config import opt


@torch.no_grad()
def predict(model_path, test_path, dst_path, transforms):

    device = torch.device("cpu")

    model = tv.models.segmentation.fcn_resnet101(pretrained=False, progress=False, num_classes=1).eval()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    for path in glob(os.path.join(test_path, '*.tif')):
        name = os.path.basename(path)
        print(name)
        img = cv2.imread(path)
        shape = (img.shape[1], img.shape[0])
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        X = transforms(img).to(device)
        X.unsqueeze_(0)
        Y = torch.sigmoid(model(X)['out'])
        output = Y.squeeze().cpu().numpy()
        output = cv2.resize(output, shape, interpolation=cv2.INTER_AREA) > 0.5
        output = (output*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_path, name), output)


if __name__ == "__main__":
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model_path = "checkpoints/model_100.pth"
    test_path = "test"
    dst_path = "results"

    predict(model_path, test_path, dst_path, transforms)
