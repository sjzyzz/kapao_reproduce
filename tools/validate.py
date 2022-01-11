import os.path as osp
from pathlib import Path

import numpy as np

from lib.general import non_max_suppression_kp, scale_coords


def run_nms(data, model_out):
    if data['iou_thres'] == data[
            'iou_thres_kp'] and data['conf_thres'] <= data['conf_thres_kp']:
        dets = non_max_suppression_kp(model_out,
                                      data['conf_thres'],
                                      data['iou_thres'],
                                      num_coords=data['num_coords'])
        person_dets = [d[d[:, 5] == 0] for d in dets]
        kp_dets = [d[data['conf_thres_kp'] <= d[:, 4]] for d in dets]
        kp_dets = [d[0 < d[:, 5]] for d in kp_dets]
    else:
        person_dets = non_max_suppression_kp(model_out,
                                             data['conf_thres'],
                                             data['iou_thres'],
                                             classes=[0],
                                             num_coords=data['num_coords'])
        kp_dets = non_max_suppression_kp(model_out,
                                         data['conf_thres_kp'],
                                         data['iou_thres_kp'],
                                         classes=list(
                                             range(1,
                                                   1 + len(data['kp_flip']))),
                                         num_coords=data['num_coords'])
    return person_dets, kp_dets


def post_process_batch(data,
                       imgs,
                       paths,
                       shapes,
                       person_dets,
                       kp_dets,
                       two_stage=False,
                       pad=0,
                       device='cpu',
                       model=None,
                       origins=None):
    '''
    In fact, i even donot know the meaning of `path`, `two_stage`, `pad`, `device`, `model` and `origins`.
    And they are not used in `image.py`. I donot know where this code if copied from lmao...
    '''
    batch_bboxes, batch_poses, batch_scores, batch_ids = [], [], [], []
    n_fused = np.zeros(data['num_coords'] // 2)

    # ok this thing is fuck
    if origins is None:
        # used only for two stage inference so set to 0 if None
        origins = [np.array([0, 0, 0]) for _ in range(len(person_dets))]

    for si, (person_det, kp_det,
             origin) in enumerate(zip(person_dets, kp_dets, origins)):
        num_person_det = person_det.shape[0]
        num_kp_det = kp_det.shape[0]

        if num_person_det:
            # the fucking `path` is useless too, motherfucker!
            # what the hell is this code copied from lmao
            # TODO: though this does not matter, but i just donot know why set `shapes` as this
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
            img_id = int(osp.splitext(osp.split(path)[-1])[0]) if path else si

            if two_stage:
                # TODO: i just have not seen any use of this shit now
                #       or i even do not know what is going on in this branch
                pass
            else:
                scores = person_det[:, 4].cpu().numpy()
                bboxes = scale_coords(imgs[si].shape[1:], person_det[:, :4],
                                      shape).round().cpu().numpy()
                poses = scale_coords(imgs[si].shape[1:],
                                     person_det[:, -data['num_coords']:],
                                     shape).cpu().numpy()
                # TODO: idn why this operation
                # NOTE: i think this is add a confidence
                # TODO: why sign - here lol
                poses = poses.reshape((num_person_det, -data['num_coords'], 2))
                poses = np.concatenate(
                    (poses, np.zeros((num_person_det, poses.shape[1], 1))),
                    axis=-1)

                if data['use_kp_dets'] and num_kp_det:
                    # this line means only with confidence higher than `conf_thres_kp_person` will be used for fusion
                    mask = scores > data['conf_thres_kp_person']
                    poses_mask = poses[mask]

                    if len(poses_mask):
                        kp_det[:, :4] = scale_coords(imgs[si].shape[1:],
                                                     kp_det[:, :4], shape)
                        kp_det = kp_det[:, :6].cpu().numpy()

                        for x1, y1, x2, y2, conf, cls in kp_det:
                            x, y = np.mean((x1, x2)), np.mean((y1, y2))
                            pose_kps = poses_mask[:, int(cls - 1)]
                            dist = np.linalg.norm(pose_kps[:, :2] -
                                                  np.array([[x, y]]),
                                                  axis=-1)
                            kp_match = np.argmin(dist)
                            if pose_kps[kp_match, 2] < conf and dist[
                                    kp_match] < data['overwrite_tol']:
                                pose_kps[kp_match] = [x, y, conf]
                                n_fused[int(cls - 1)] += 1

                        poses[mask] = poses_mask
                poses = [p + origin for p in poses]

            batch_bboxes.extend(bboxes)
            batch_poses.extend(poses)
            batch_scores.extend(scores)
            batch_ids.extend([img_id] * len(scores))

    return batch_bboxes, batch_poses, batch_scores, batch_ids, n_fused
