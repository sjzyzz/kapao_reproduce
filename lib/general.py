import math
import time

import numpy as np
import torch
import torchvision

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32, floor=0):
    '''
    Vertify and adjust to make sure that the image size is a multiple of stride s
    '''
    if isinstance(img_size, int):
        new_size = max(make_divisible(img_size, int(s)), floor)
    else:
        # in fact, i even dont know what this represent
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    if new_size != img_size:
        print(f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
    return new_size

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''
    rescale coords from img1_shape to img0_shape
    '''
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2 # w, h
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    nl = coords.shape[0]
    # TODO: why use reshape???
    # NOTE: because there are boxes, which have two coordinates, and they need to be treat equally
    coords = coords.reshape((nl, -1 ,2))
    coords[..., 0] -= pad[0]
    coords[..., 1] -= pad[1]
    coords /= gain
    # TODO: and why reshape back???
    # NOTE: just see above
    coords = coords.reshape(nl, -1)
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, img_shape[1])
        boxes[:, 1].clamp_(0, img_shape[0])
        boxes[:, 2].clamp_(0, img_shape[1])
        boxes[:, 3].clamp_(0, img_shape[0])
    else: # np.array
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def non_max_suppression_kp(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300, num_coords=34):
    nc = prediction.shape[2] - 5 - num_coords
    xc = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 4096
    max_nms = 30000 # the max number send to `torch.nms` func
    time_limit = 10.0
    redundant = True
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6 + num_coords), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]] # use confidence as filter

        if not x.shape[0]:
            continue
            
        x[:, 5:-num_coords] *= x[:, 4:5] # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:, :4])

        # wth is this fucking `dim` man
        conf, j = x[:, 5:-num_coords].max(1, keepdim=True)
        kp = x[:, -num_coords:]
        x = torch.cat((box, conf, j.float(), kp), 1)[conf_thres < conf.view(-1)]

        # filter by class
        if classes is not None:
            # TODO: i know what he want to do but just dont know how he did that shit
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        n = x.shape[0]
        if not n:
            continue
        elif max_nms < n:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if max_det < i.shape[0]:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            # Merge NMS (boxes merged using weighted mean)
            # TODO: to be honest, idn what is going on here
            pass
        
        output[xi] = x[i]
        if time_limit < (time.time() - t):
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    
    return output
        