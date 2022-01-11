from contextlib import contextmanager
import logging
from copy import deepcopy
import math
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

LOGGER = logging.getLogger(__name__)
LOGGING_LEVEL = logging.INFO if os.getenv('RANK', -1) in [-1, 0] else logging.WARN
LOGGER.setLevel(LOGGING_LEVEL)


def select_device(device=''):
    '''
    return the specific device, if `device == ''`, it will return `cuda:0` device
    '''
    if device == 'cpu':
        print('Using CPU now, the speed is pretty slow, hope u know that lol')
        return torch.device('cpu')
    elif device == '':
        print('As the default case, we just use cuda:0')
        return torch.device('cuda:0')
    else:
        device = str(device).strip()
        return torch.device(f"cuda:{device}")


def model_info(model, verbose=False, img_size=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters())
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s % 10s" %
            ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma')
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            # TODO: idn why this operation
            name = name.replace('module_list', '')
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g" % (
                    i, name, p.requires_grad, p.numel(), list(p.shape),
                    p.mean(), p.std()
                )
            )

    # TODO: i just dont want to care FLOPs now lol
    fs = ''

    # TODO: wht is `model.modules` lol
    LOGGER.info(
        f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}'
    )


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel, nn.parallel.DistributedDataParallel
    )


# TODO: i know that this function just let all other processes wait for the main process,
#       but idn how it is implemented
# NOTE: the code before `yield` will be executed when this function is called in `with` statement.
#       And after the code in `with` is done, the code after `yield` will be executed.
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def intersect_dicts(da, db, exclude=()):
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k
                               for x in exclude) and v.shape == db[k].shape
    }


def de_parallel(model):
    return model.module if is_parallel(model) else model

def copy_attr(a, b, include=(), exclude=()):
    """
    Copy attributes from b to a
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """
    Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model
                           ).eval()
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            # TODO: idn why use this `d` to decay but not directly use original `decay`
            d = self.decay(self.updates)

            msd = model.module.state_dict(
            ) if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
    
    def updata_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        Update EMA attributes
        """

        copy_attr(self.ema, model, include, exclude)


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:
        # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

