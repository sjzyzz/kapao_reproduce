import logging

import torch
import torch.nn as nn

from lib.general import get_logger

from utils.metrics import bbox_iou
from utils.loss import smooth_BCE, FocalLoss

from lib.torch_utils import is_parallel

logging.basicConfig(level=logging.DEBUG)
LOGGER = get_logger(__name__)


class MyComputeLoss:
    def __init__(self, model, autobalance=False, num_coords=0):
        super(MyComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device)
        )

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        g = h['fl_gamma']
        if 0 < g:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.balance = {
            3: [4.0, 1.0, 0.4]
        }.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        if self.autobalance:
            self.loss_coeffs = model.module.loss_coeffs if is_parallel(
                model
            ) else model.loss_coeffs[-1]

        self.num_coords = num_coords
        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors

    def __call__(self, prediction, targets, all_mine=True):

        loss_obj = torch.zeros(1, device=targets.device)
        loss_cls = torch.zeros(1, device=targets.device)
        loss_box = torch.zeros(1, device=targets.device)
        loss_kp = torch.zeros(1, device=targets.device)

        if all_mine:
            tcls, tbox, tkp, indices, anchors = self.my_build_targets_lol(
                prediction, targets
            )
        else:
            tcls, tbox, tkp, indices, anchors = self.build_targets(
                prediction, targets
            )

        device = targets.device
        for i, prediction_i in enumerate(prediction):
            img_idx, anchor_idx, row_idx, column_idx = indices[i]
            tobj = torch.zeros_like(prediction_i[..., 4], device=device)
            num_targets = img_idx.shape[0]
            if num_targets:
                responsible_prediction_i = prediction_i[img_idx, anchor_idx,
                                                        row_idx, column_idx]
                pxy = responsible_prediction_i[:, :2].sigmoid() * 2 - 0.5
                pwh = (responsible_prediction_i[:, 2:4].sigmoid() *
                       2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), dim=-1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)

                # objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    pass
                # TODO: idn why it is `score`, in my opinion, it should be `1` lol
                tobj[img_idx, anchor_idx, row_idx,
                     column_idx] = (1.0 - self.gr) + self.gr * score_iou

                # class
                if 1 < self.nc:
                    t = torch.full_like(
                        responsible_prediction_i[:, 5:5 + self.nc],
                        self.cn,
                        device=device
                    )
                    t[range(num_targets), tcls[i]] = self.cp
                    loss_cls += self.BCEcls(
                        responsible_prediction_i[:, 5:5 + self.nc], t
                    )

                # box
                loss_box += (1.0 - iou).mean()

                # keypoints
                if self.num_coords:
                    tkp_i = tkp[i]
                    is_pose_mask = 0 < tkp_i[..., 2]
                    t_pose_kp_i = tkp_i[is_pose_mask]
                    if len(t_pose_kp_i):
                        p_all_kp_i = responsible_prediction_i[
                            ..., 5 + self.nc:].reshape(
                                (-1, self.num_coords // 2, 2)
                            )
                        p_all_kp_i = (p_all_kp_i.sigmoid() * 4 -
                                      2) * anchors[i][:, None, :]
                        p_pose_kp_i = p_all_kp_i[is_pose_mask]
                        # TODO: in fact, idn why `dim=-1`
                        l2 = torch.linalg.norm(
                            p_pose_kp_i - t_pose_kp_i[..., :2], dim=-1
                        )
                        loss_kp += torch.mean(l2)

            loss_obj_i = self.BCEobj(prediction_i[..., 4], tobj)
            loss_obj += loss_obj_i * self.balance[i]

        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']
        loss_box *= self.hyp['box']
        loss_kp *= self.hyp['kp']

        if self.autobalance:
            pass
        else:
            loss = loss_obj + loss_cls + loss_box + loss_kp

        batch_size = prediction[0].shape[0]

        return loss * batch_size, torch.cat(
            (loss_box, loss_obj, loss_cls, loss_kp)
        ).detach()

    def my_build_targets_lol(self, prediction, targets):
        num_anchor, num_target = self.na, targets.shape[0]
        indices, classes, boxes, anchors, kps = [], [], [], [], []
        multipier = torch.ones(
            1 + 1 + 4 + (self.num_coords // 2) * 3 + 1, device=targets.device
        )
        anchor_idx = torch.arange(num_anchor, device=targets.device).view(
            num_anchor, 1
        ).repeat(1, num_target)
        targets = torch.cat(
            (targets.repeat(num_anchor, 1, 1), anchor_idx[..., None]), dim=2
        )

        g = 0.5
        # TODO: why times `g` lol
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ], device=targets.device
        ).float() * g
        # for each level
        for i in range(self.nl):
            anchors_i = self.anchors[i]
            wh = torch.tensor(prediction[i].shape,
                              device=targets.device)[[3, 2]]
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
                ratio_mask = torch.max(ratio, 1.0 /
                                       ratio).max(2)[0] < self.hyp['anchor_t']
                targets_i = targets_i[ratio_mask]

                # next, we need to find for every target, which grids are responsible
                gxy = targets_i[:, 2:4]
                gxy_inverse = wh - gxy
                left, top = ((gxy % 1. < g) & (1. < gxy)).T
                right, bottom = ((gxy_inverse % 1. < g) & (1. < gxy_inverse)).T
                mask = torch.stack(
                    (torch.ones_like(left), left, top, right, bottom)
                )
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

            anchor_idx = targets_i[:, -1].long()
            indices.append(
                (
                    img_idx, anchor_idx, gj.clamp_(0, wh[1] - 1),
                    gi.clamp_(0, wh[0] - 1)
                )
            )
            anchors.append(anchors_i[anchor_idx])
            classes.append(cls)
            boxes.append(torch.cat((gxy - gij, gwh), dim=1))
        return classes, boxes, kps, indices, anchors