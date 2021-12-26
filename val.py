
from lib.general import non_max_suppression_kp

def run_nms(data, model_out):
    if data['iou_thres'] == data['iou_thres_kp'] and data['conf_thres'] <= data['conf_thres_kp']:
        dets = non_max_suppression_kp(model_out, data['conf_thres'], data['iou_thres'], num_coords=data['num_coords'])
        person_dets = [d[d[:, 5] == 0] for d in dets]
        kp_dets = [d[data['conf_thres_kp'] <= d[:, 4]] for d in dets]
        kp_dets = [d[0 < d[:, 5]] for d in kp_dets]
    else:
        person_dets = non_max_suppression_kp(model_out, data['conf_thres'], data['iou_thres'], classes=[0], num_coords=data['num_coords'])
        kp_dets = non_max_suppression_kp(model_out, data['conf_thres_kp'], data['iou_thres_kp'], classes=list(range(1, 1 + len(data['kp_flip']))), num_coords=data['num_coords'])
    return person_dets, kp_dets