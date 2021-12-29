import glob
import os
import os.path as osp
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.augmentations import letterbox
from lib.torch_utils import torch_distributed_zero_first


def create_dataloader(
    path,
    labels_dir,
    img_size,
    batch_size,
    stride,
    single_cls=False,
    hyp=False,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix='',
    kp_flip=None,
    kp_bbox=None,
):
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            labels_dir,
            img_size,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            kp_flip=kp_flip,
            kp_bbox=kp_bbox,
        )

    batch_size = min(batch_size, len(dataset))
    num_workers = min(
        [os.cpu_count(), batch_size if 1 < batch_size else 0, workers]
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset
    ) if rank != -1 else None
    loader = torch.utils.data.Dataloader if image_weights else InfiniteDataLoader
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4
        if quad else LoadImagesAndLabels.collate_fn
    )
    return dataloader, dataset


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
        img0 = cv2.imread(path)  # idn but it seems in BGR format
        # idn why use this assertion but it seems the problem of `cv2.imread`
        assert img0 is not None, 'Image Not Found' + path
        print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(
            img0, self.img_size, stride=self.stride, auto=self.auto
        )[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC 2 CHW, BGR 2 RGB
        img = np.ascontiguousarray(
            img
        )  # i just know this stuff, u cant draw box on image with contiguous

        img = torch.from_numpy(img)
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]

        return path, img, img0


class LoadImagesAndLabels(Dataset):
    def __init__():
        pass

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            pass
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]
                                     ] if self.rect else self.img_size
            img, ratio, pad = letterbox(
                img, shape, auto=False, scaleup=self.augment
            )
            # TODO: in fact, idn the use of this
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:],
                    ratio[0] * w,
                    ratio[1] * h,
                    padw=pad[0],
                    padh=pad[1],
                )

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp['degrees'],
                    translate=hyp['translate'],
                    scale=hyp['scale'],
                    perspective=hyp['perspective'],
                    kp_bbox=self.kp_bbox,
                )

        nl = len(labels)
        if nl:
            labels[:, 1:] = xyxy2xywhn(
                labels[:, 1:],
                w=img.shape[1],
                h=img.shape[0],
                clip=True,
                eps=1E-3,
            )

        if self.augment:
            img, lables = self.albumentations(img, labels)
            nl = len(labels)

            # HSV color-space
            augment_hsv(
                img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v']
            )

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    lables[:, 1] = 1 - lables[:, 1]

                    if self.kp_flip and 5 < lables.shape[1]:
                        # flip keypoints in pose object
                        labels[:, 5::3] = 1 - labels[:, 5::3]
                        keypoints = labels[:, 5:].reshape(nl, -1, 3)
                        keypoints = keypoints[:, self.kp_flip
                                             ]  # reorder left-right keypoints
                        lables[:, 5:] = keypoints.reshape(nl, -1)

                    if self.obj_flip:
                        for i, cls in enumerate(labels[:, 0]):
                            labels[i, 0] = self.obj_flip[labels[i, 0]]

        labels_out = torch.zeros((nl, labels.shape[-1] + 1))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes
