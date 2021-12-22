import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    '''
    Resize and pad image while meeting stride-multiple constraints
    '''
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: # idn what the hell is this fucking stuff doing, but it only scale down
        r = min(r, 1.0)
    
    # let's get the padding
    ratio = r, r # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)), 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        # idn what this fucking arugment meaning lol
        # just as below, add border and maintain the width-height ratio
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill: 
        # idn this fucking argument either lmao
        # lol i know now, this argument means only scale (i.e. without border) to `new_shape`
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # TODO: the detail here may need more attention lol
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)