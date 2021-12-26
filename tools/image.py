import argparse
import os.path as osp
import yaml

import cv2

import _init_path
from lib.torch_utils import select_device
from lib.general import check_img_size
from lib.datasets import LoadImages
from lib.yolo import Model
from lib.experimental import load_weights
from lib.general import scale_coords
from val import run_nms#, post_process_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get lots of arguments
    parser.add_argument('--cfg', type=str, default='cfg/yolov5s6.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--weights', default='kapao_l_coco.pt')
    parser.add_argument('--img-size', type=int, default=1280)
    parser.add_argument('--img-path', default='res/crowdpose_100024.jpg', help='path to image')
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--color-pose', type=int, nargs='+', default=[255, 0, 255], help='pose object color')
    parser.add_argument('--line-thick', type=int, default=2, help='line-thick')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    # just add arguments below to the `data` dict to do the inference, at least i think so
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--iou-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint object to do the fusion')
    # TODO: in fact, you just dont know what this arguments doing
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--overwrite-tol', type=int, default=25)
    args = parser.parse_args()
    # print(args)

    # use yaml file to save the info of the data
    with open(args.data) as f:
        data = yaml.safe_load(f)
    # add inference setting to data dict
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['iou_thres'] = args.iou_thres
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres'] = args.conf_thres
    data['conf_thres_kp'] = args.conf_thres_kp
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    # first select which gpu
    device = select_device(args.device)
    print(f"Using device: {device}")
    # time to get the model lol
    # NOTE: basically, in the initialization of model,
    #       if the `anchors` is specified, the model will use the specified `anchors`,
    #       or it will use the `anchors` in `cfg` file
    model = Model(args.cfg, ch=3, nc=data['nc'], anchors=None, num_coords=data['num_coords']).to(device)
    model = load_weights(args.weights, model, device)
    model.eval()
    stride = int(model.stride.max())
    img_size = check_img_size(args.img_size, s=stride)
    dataset = LoadImages(args.img_path, img_size=img_size, stride=stride, auto=True)

    (_, img, img0) = next(iter(dataset))
    img = img.to(device)
    
    # TODO: did not use augment here, just add it later
    out = model(img, augment=False, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]

    bbox_dets, kp_dets = run_nms(data, out)

    if args.bbox:
        # TODO:the index here is wire
        bboxes = scale_coords(img.shape[2:], bbox_dets[0][:, :4], img0.shape[:2]).round().detach().cpu().numpy()
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_pose, thickness=args.line_thick)
    
    # i think this is the fusion of pose object and kp object
    # but the interface is really ugly lol
    # _, poses, _, _, _ = post_process_batch(data, img, [], [[img0.shape[:2]]], [x.detach() for x in bbox_dets], [x.detach() for x in kp_dets])
    # if args.pose:
    #     for pose in poses:
    #         for seg in data['segments'].values():
    #             # really don't know what is going on here, that's really normal lol
    #             pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
    #             pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
    #             cv2.line(img0, pt1, pt2, args.color_pose, args.line_thick)
    
    # TODO: here comes two method `splitext` and `split`, u need to figure out the usage of these two
    filename = '{}_{}'.format(osp.splitext(osp.split(args.img_path)[-1])[0], osp.splitext(args.weights)[0])
    if args.bbox:
        filename += '_bbox'
    if args.pose:
        filename += '_pose'
    filename += '.png'
    cv2.imwrite(filename, img0)