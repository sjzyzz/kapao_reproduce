import torch

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