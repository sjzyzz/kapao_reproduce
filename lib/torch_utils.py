import logging

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


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
        print("%5s %40s %9s %12s %20s %10s % 10s" %
              ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu',
               'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            # TODO: idn why this operation
            name = name.replace('module_list', '')
            print("%5g %40s %9s %12g %20s %10.3g %10.3g" %
                  (i, name, p.requires_grad, p.numel(), list(
                      p.shape), p.mean(), p.std()))

    # TODO: i just dont want to care FLOPs now lol
    fs = ''

    # TODO: wht is `model.modules` lol
    LOGGER.info(
        f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}'
    )

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
