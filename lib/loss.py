import logging

import torch
import torch.nn as nn

from utils.metrics import bbox_iou

from torch_utils import is_parallel

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

class ComputeClass:
    def __init__():
        pass
    
    def __call__(self, prediction, targets):
        pass

    def _build_target(self, prediction, targets):
        num_anchor, num_target = self.num_anchor, targets.shape[0]
        indices, classes, boxes, anchors, kps = [], [], [], [], []
        multipier = torch.ones(1 + 1 + 4 + (self.num_coords // 2) * 3 + 1, device=self.device)
        anchor_idx = torch.arange(self.num_anchor).view(num_anchor, 1).repeat(1, num_target)
        targets = torch.cat((targets.repeat(num_anchor, 1, 1), anchor_idx[:, None]), dim=2)

        g = 0.5
        # TODO: why times `g` lol
        off = torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        ], device=targets.device).float() * g
        # for each level
        for i in range(self.nl):
            anchors_i = self.anchors[i]
            wh = torch.tensor(prediction[i].shape)[[3, 2]]
            multipier[2:4] = wh
            multipier[4:6] = wh
            for j in range(self.num_coords // 2):
                start_idx = 6 + j * 3
                multipier[start_idx:start_idx + 2] = wh
            
            # scale the target to level space
            targets_i = targets * multipier
            if num_target:
                # use wh ratio with anchor as a filter
                ratio = targets_i[:, :, 4:6] / anchors_i[:, None]
                # TODO: the `max` func here need more consideration
                ratio_mask = torch.max(ratio, 1.0 / ratio).max(2)[0] < self.hyp['anchor_t']
                targets_i = targets_i[ratio_mask]

                # next, we need to find for every target, which grids are responsible
                gxy = targets[:, 2:4]
                gxy_inverse = wh - gxy
                left, top = ((gxy % 1. < g) & (1. < gxy)).T
                right, bottom = ((gxy_inverse % 1.) & (1. < gxy_inverse)).T
                mask = torch.stack((torch.ones_like(left), left, top, right, bottom))
                targets_i = targets_i.repeat((5, 1, 1))[mask]
                # TODO: the index here need more consideration
                offset = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
            else:
                # TODO: idn what to do here
                targets_i = targets[0]
                offset = 0
            
            img_idx = targets_i[:, 0].long()
            cls = targets_i[:, 1].long()
            gxy = targets_i[:, 2:4]
            gwh = targets_i[:, 4:6]
            # TODO: now there are many duplicate elements, basically 3x, need to do some operation;)
            # NOTE: it is right, the relative to the corresponding grid should be subtracted the offset,
            #       for instance, the left should substract by 1 in x dim.
            # TODO: but why multiple the `g` lmao, and i think the result should be the same if `g == 1`.
            #       maybe you can do some experience
            gij = (gxy - offset).long()
            gi, gj = gij.T

            if self.num_coords:
                kp = targets_i[:, 6:-1].reshape(-1, self.num_coords // 2, 3)
                kp[..., :2] -= gij[:, None, :]
                kps.append(kp)

            anchor_idx = targets_i[:, -1]
            indices.append((img_idx, anchor_idx, gj.clamp_(0, wh[1] - 1), gi.clamp_(0, wh[0] - 1)))
            anchors.append(anchors_i[anchor_idx])
            classes.append(cls)
            boxes.append(torch.cat((gxy - gij, gwh), dim=1))
        return classes, boxes, kps, indices, anchors

