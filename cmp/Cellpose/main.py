import os
import cv2
import tifffile
import numpy as np
from glob import glob
from cellpose import models

model = models.Cellpose(model_type='cyto')

if __name__ == "__main__":
    test_path = "test"
    dst_path = "results"
    ele = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for path in glob(os.path.join(test_path, '*.tif')):
        name = os.path.basename(path)
        img = [cv2.imread(path)]
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=[[2, 0]])
        img = masks[0]
        maxId = np.amax(img) + 1

        res = np.zeros_like(img).astype(np.uint8)
        for i in range(1, maxId):
            tmp = (img == i) * 255
            res |= cv2.erode(tmp.astype(np.uint8), ele)

        res = (res == 0) * 255
        cv2.imwrite(os.path.join(dst_path, name), res.astype(np.uint8))
