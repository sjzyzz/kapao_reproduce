import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

from utils.torch_utils import time_sync, scale_img, initialize_weights
from utils.plots import feature_visualization
from utils.autoanchor import check_anchor_order

from general import make_divisible
from common import *


try:
    import thop
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class Detect(nn.Module):
    stride=None

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True, num_coords=0):
        super.__init__()
        self.nc = nc # number of classes
        self.no = nc + 5 # number of outputs per anchor
        self.nl = len(anchors) # number of layers
        self.na = len(anchors[0]) // 2 # number of anchors per layer
        self.grid = [torch.zeros(1)] * self.nl
        # NOTE: u just have not use this argument yet, so dont worry about it lol
        #       u know the reason why set this stuff like this now lol
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        #   the buffer is part of the module which means it will be saved to state_dict
        #   but it is not the module's parameter which means its `require_grad == False`
        self.register_buffer('anchors', a) # (nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2)) # (nl, 1, na, 1, ,1 ,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace
        # TODO: idn what the hell is this fucking argument is setting for
        self.num_coords = num_coords
    
    def forward(self, x):
        z = []
        for i in range(self.nl):
            # for each layer(for multi-scale prediction)
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape # from (bs, 255, 20, 20) -> (bs, num_anchors, h, w, num_output)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    # NOTE: idn what this means or why use this argument
                    #       lol pretty easy stuff man, 
                    #       it just do the in-place operation which means there is no need to allocate new space

                    # this two stuff is to do the transformation for xy and wh
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    # NOTE: what the hell is `anchor_grid` lmao
                    #       fine, just use the broadcast mechanism to multiply the width and height of the original anchor
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]

                    if hasattr(self, 'num_coords') and self.num_coords:
                        # TODO: what the hell is this shit?
                        pass
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    # TODO: can anyone tell me what the hell is `dim` func lol, 
                    #       it drives me crazy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # TODO: in fact, i just dont know why use view func, and why the `dim` in 
        #       `stack` func is 2, i mean, how to understand the `dim` argument lol?
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, num_coords=0, autobalance=False):
        # TODO: here is a question, what the hell is `self.module`
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            # TODO: what `.name` means
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)
        
        # define model
        # TODO: what the hell is this code fucking doing, why not just get the channel lmao
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc + num_coords and nc + num_coords != self.yaml['nc']:
            # TODO: idn why there is an error lmao
            # LOGGER.info(f'Overriding model.yaml nc={self.yaml['nc']} with nc={nc + num_coords}')
            self.yaml['nc'] = nc + num_coords
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)
        # TODO: idn what `autobalance` means
        if autobalance:
            self.loss_coeffs = nn.parameter(torch.zeros(2))

        # build strides, anchors
        m = self.model[-1] # you get Detect() now
        if isinstance(m, Detect):
            # TODO: what the hell is this doing?
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.num_coords = self.num_coords
            m.nc = nc
            self._initialize_biases()
        
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualiza=False, kp_flip=None, scales=[0.5, 1, 2], flip=[None, 3, None]):
        if augment:
            return self.forward_augment(x, kp_flip, s=scales, f=flip)
        return self.forward_once(x, profile, visualiza)
    
    def forward_augment(self, x, kp_flip, s=[0.5, 1, 2], f=[None, 3, None]):
        # TODO: idn what the hell is this fucking stuff doing lol
        img_size = x.shape[-2:] # h, w

        y = []
        train_out = None
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi, train_out_i = self.forward_once(xi)
            if si == 1 and fi is None:
                train_out = train_out_i
            yi = self._descale_pred(yi, fi, si, img_size, kp_flip)
            y.append(yi)
        return torch.cat(y, 1), train_out
    
    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            # TODO: lol idn what the hell this if statement doing
            # NOTE: the `Detect` need a list as input, which means multi-scale predicton
            if m.f != -1: 
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] # from earlier layers
            
            if profile:
                # i think this argument is for logging the model
                c = isinstance(m, Detect)
                # TODO: this dude is for computing FLOP, but i dont know that much
                o = thop.profile(m, inputs=(x.copy() if c else x), verbose=False)[0] / 1E9 * 2 if thop else 0
                # TODO: man this operation is too fucking high, i even dont know why
                # NOTE: fine, it will just get a time, but because of there are multiple thread running
                #       at the same time, so maybe they need to synv lol
                t = time_sync()
                # TODO: why do 10 times lol
                # NOTE: i think this statement is just run 10 times to get a stabler value
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'param':>10s} {'module'}")
                LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f} {m.type}")
            
            x = m(x)
            y.append(x if m.i in self.save else None)

            if visualize:
               feature_visualization(x, m.type, m.i, save_dir=visualize) 
            
        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x
    
def parse_model(d, ch):
    '''
    Given the model_dict and input_channels, return the full model
    '''
    LOGGER.info('\n%3s%18s%3s%10s   %-40s%-30s' % ('', 'from', 'n', 'param', 'module', 'arguments'))
    # TODO: idn why setting `depth_multiple` and `width_multiple`
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # TODO: i think the last `anchors` should change to `len(anchors) // 2`
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']): # from, number, module, args
        # TODO: idn why use this code
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        
        # TODO: idn what this code doing
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, Bottleneck, SPP, Focus, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                # TODO: idn why this code
                c2 = make_divisible(c2 * gw, 8)
            
            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)
                n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]
            
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if 1 < n else m(*args)
            t = str(m)[8:-2].replace('__main__', '')
            np = sum([x.numel() for x in m_.parameters()])
            m_.i, m_.f, m_.type, m_.np = i, f, t, np
            LOGGER.info('%3s%18s%3s%10.0f   %-40s%-30s' % (i, f, n_, np, t, args))
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)
