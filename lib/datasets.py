import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lib.torch_utils import torch_distributed_zero_first

def create_dataloader(
    path,
    labels_dir,
    img_size,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
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
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn 4 if quad else LoadImagesAndLabels.collate_fn
    )
    return dataloader, dataset

class LoadImagesAndLabels(Dataset):
    
    def __init__(self, path, labels_dir='labels', img_size=640, batch_size=16, augment=False, hyp=None, rect=False,image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', kp_flip=None, kp_bbox=None,):
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment=augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        self.kp_flip = kp_flip
        self.kp_bbox = kp_bbox
        self.num_coords = len(kp_flip) * 2

        if self.kp_flip:
            pass
        else:
            self.obj_flip = None

        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)