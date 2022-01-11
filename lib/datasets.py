import glob
import random
import logging
import os
from pathlib import Path
from multiprocessing import Pool
from numpy.core.fromnumeric import repeat

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.datasets import Albumentations, InfiniteDataLoader, IMG_FORMATS, HELP_URL, img2label_paths, get_hash, load_image, verify_image_label, letterbox
from utils.general import xywhn2xyxy, xyxy2xywhn

from lib.torch_utils import torch_distributed_zero_first

NUM_THREADS = 4


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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset
    ) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4
        if quad else LoadImagesAndLabels.collate_fn
    )
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):
    def __init__(
        self,
        path,
        labels_dir='labels',
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix='',
        kp_flip=None,
        kp_bbox=None,
    ):
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
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

        # through the code block below, you get all the file path in relative format
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace('./', parent) if x.startswith('./') else x for x in t
                        ]
                else:
                    raise Exception(f'{prefix}{p} dose not exist')

            self.img_files = sorted(
                [
                    x.replace('/', os.sep)
                    for x in f if x.split('.')[-1].lower() in IMG_FORMATS
                ]
            )
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(
                f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}'
            )

        # Check cache
        self.label_files = img2label_paths(
            self.img_files, labels_dir=self.labels_dir
        )
        # TODO: idk the meaning of `p` here
        cache_path = (p if p.is_file() else Path(self.label_files[0]
                                                ).parent).with_suffix('.cache')

        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache['version'] == 0.4 and cache[
                'hash'] == get_hash(self.label_files + self.img_files)
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            # TODO: idk what this meaning
            tqdm(None, desc=prefix + d, total=n, initial=n)
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))
        assert nf > 0 or not augment, f'{prefix}No label in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ['hash', 'version', 'msgs']]
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys(), labels_dir=labels_dir)
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)

        if self.rect:
            pass

        # Cache images into memory for faster training
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            pass

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """
        Cache dataset labels, and check images and read shapes
        """
        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    verify_image_label,
                    zip(
                        self.img_files, self.label_files, repeat(prefix),
                        repeat(self.num_coords)
                    )
                ),
                desc=desc,
                total=len(self.img_files)
            )
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        # TODO: idk why we need to close...
        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(
                f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}'
            )
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['imgs'] = msgs
        x['version'] = 0.4
        # Save the result to path
        try:
            np.save(path, x)
            path.with_suffix('.cache.npy').rename(path)
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(
                f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}'
            )
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Returns:
            torch.Tensor:
                with shape C x H x W, C is in RGB order
            torch.Tensor:
                with shape label_num x label_length, (img_idx, x, y, w, h, cls), where img_idx is determined in training, 
                x, y, w and h has been normalized
            str:
                realative path of this image
            ???:
                just something I have not used yet
        """

        index = self.indices[index]

        hyp = self.hyp
        # TODO: maybe one day add mosaic function
        # mosaic = self.mosaic and random.random() < hyp['mosaic']
        mosaic = False
        if mosaic:
            pass
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            shape = self.batch_shape[self.batch[index]
                                    ] if self.rect else self.img_size
            img, ratio, pad = letterbox(
                img, shape, auto=False, scaleup=self.augment
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:],
                    ratio[0] * w,
                    ratio[1] * h,
                    padw=pad[0],
                    padh=pad[1]
                )

            if self.augment:
                pass

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
            pass

        labels_out = torch.zeros((nl, labels.shape[-1] + 1))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes