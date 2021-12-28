import torch
import torch.nn as nn

from utils.metrics import bbox_iou


class ComputeLoss:
    def __init__(self, model, autobalance=False, num_coords=0):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device
        h = model.hyp
        
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.balance = {3:[4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        if self.autobalance:
            self.loss_coeffs = model.module.loss_coeffs if is_parallel(model) else model.loss_coeffs[-1]
        
        self.num_coords = num_coords
        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors

    def __call__(self, p, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lkps = torch.zeros(1, device=device)
        tcls, tbox, tkps, indices, anchors = self._build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(p[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                if self.num_coords:
                    tkp = tkps[i]
                    vis = tkp[..., 2] > 0
                    tkp_vis = tkp[vis]
                    if len(tkp_vis):
                        pkp = ps[:, 5 + self.nc:].reshape(
                            (-1, self.num_coords // 2, 2))
                        pkp = (pkp.sigmoid() * 4. - 2.) * anchors[i][:,
                                                                     None, :]
                        pkp_vis = pkp[vis]
                        l2 = torch.linalg.norm(pkp_vis - tkp_vis[..., :2],
                                               dim=-1)
                        lkps += torch.mean(l2)

                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[
                        sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:5 + self.nc],
                                        self.cn,
                                        device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5 + self.nc], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkps *= self.hyp['kp']

        if self.autobalance:
            loss = (lbox + lobj + lcls) / (torch.exp(
                2 * self.loss_coeffs[0])) + self.loss_coeffs[0]
            loss += lkps / (torch.exp(
                2 * self.loss_coeffs[1])) + self.loss_coeffs[1]
        else:
            loss = lbox + lobj + lcls + lkps

        bs = tobj.shape[0]

        return loss * bs, torch.cat((lbox, lobj, lcls, lkps)).detach()

    def _build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, tkps, indices, anch = [], [], [], [], []
        gain = torch.ones(7 + self.num_coords * 3 // 2, device=targets.device)
        ai = torch.arange(na,
                          device=targets.device).float().view(na,
                                                              1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                           device=targets.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            xy_gain = torch.tensor(p[i].shape)[[3, 2]]
            gain[2:4] = xy_gain
            gain[4:6] = xy_gain
            for j in range(self.num_coords // 2):
                kp_idx = 6 + j * 3
                gain[kp_idx:kp_idx + 2] = xy_gain

            t = targets * gain
            if nt:
                # Match
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                t = j[t]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                # TODO: idn why this code
                # NOTE: finally!!! the j means whether the left grid is also 'responsible' to the object, the same for others
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b = t[:, 0].long()  # image
            c = t[:, 1].long()  # class
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            if self.num_coords:
                kp = t[:, 6:-1].reshape(-1, self.num_coords // 2, 3)
                kp[..., :2] -= gij[:, None, :]
                tkps.append(kp)

            a = t[:, -1].long()
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, tkps, indices, anch
