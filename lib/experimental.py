import torch

from lib.yolo import Detect, Model

# TODO: You just load one model while the original code load multiple weights
#       and you still dont know the meaning of `inplace` and `fuse`
#       seems `fuse` will determine whether do the layer fuse, 
#       so what is layer fuse lol
def attempt_load(weights, map_location=None, inplace=True, fuse=True):    

    ckpt = torch.load(weights, map_location=map_location)
    if fuse:
        # TODO: why we need to convert the model to `float()` lol and even `eval()`, and what is ema?
        #       lol i just dont want to give it a fuck now
        return ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
    else:
        return ckpt['ema' if ckpt.get('ema') else 'model'].float().eval()
    
    # There is also some code for compatibility, but i just dont need it now

def load_weights(weights, model, map_location=None):
    # load the weighs(there is a path) to the model
    ckpt = torch.load(weights, map_location=map_location)
    # TODO: idn why use `.float()` here lol
    ckpt_sd = ckpt['model'].float().state_dict()
    model.load_state_dict(ckpt_sd, strict=False)
    return model
