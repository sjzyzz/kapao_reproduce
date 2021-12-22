import glob
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch

from augmentations import letterbox

class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())
        if '*' in p:
            # if there is re
            files = sorted(glob.glob(p, recursive=True))
        elif osp.isdir(p):
            files = sorted(glob.glob(osp.join(p, '*')))
        elif osp.isfile(p):
            files = [p]
        else:
            raise Exception(f"ERROR: {p} does not exist")
        
        # idn, do i need to check the img/vid stuff?

        self.img_size = img_size
        self.stride = stride
        self.files = files
        self.nf = len(files)
        # idn, what the hell is this argument
        self.auto = auto

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path) # idn but it seems in BGR format
        # idn why use this assertion but it seems the problem of `cv2.imread`
        assert img0 is not None, 'Image Not Found' + path
        print(f'image {self.count}/{self.nf} {path}: ', end='')
        
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1] # HWC 2 CHW, BGR 2 RGB
        img = np.ascontiguousarray(img) # i just know this stuff, u cant draw box on image with contiguous

        img = torch.from_numpy(img)
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]

        return path, img, img0