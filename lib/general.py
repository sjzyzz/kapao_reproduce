import math

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