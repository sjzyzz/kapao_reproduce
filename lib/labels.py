import os
import os.path as osp

import numpy as np
from tqdm import tqdm

def write_kp_labels(data):
    """
    Main contradiction: the original annotation is not clear for keypoint object
    HOWTO:              create new annotation with more clear meaning
    """

    assert not osp.isdir(osp.join(data['path'], data['labels'])), 'Labels already generated. Remove or choose new name for labels'
    is_coco = 'coco' in data['path']
    if is_coco:
        from pycocotools import COCO
    else:
        # TODO: for crowdpose
        pass

    splits = [
        osp.splitext(osp.split(data[s])[-1])[0] for s in ['train', 'val', 'test'] if s in data
    ]
    annotations = [
        osp.join(data['path'], data['{}_annotations'.format(s)]) for s in ['train', 'val', 'test'] if s in data
    ]
    test_split = [
        0 if s in ['train', 'val'] else 1 for s in ['train', 'val', 'test'] if s in data
    ]
    img_txt_dir = osp.join(data['path'], data['labels'], 'img_txt')
    os.makedirs(img_txt_dir, exist_ok=True)

    for split, annot, is_test in zip(splits, annotations, test_split):
        img_txt_path = osp.join(img_txt_dir, f'{split}.txt')
        labels_path = osp.join(data['path'], f"{data['labels']}/{split if is_coco else ''}")
        if not is_test:
            os.makedirs(labels_path, exist_ok=True)
        coco = COCO(annot)
        if not is_test:
            pbar = tqdm(coco.anns.keys(), total=len(coco.anns.keys()))
            pbar.desc = f'Writing {split} labels to {labels_path}'
            for id in pbar:
                a = coco.anns[id]

                if a['image_id'] not in coco.imgs:
                    continue

                if 'train' in split:
                    if is_coco and a['iscrowd']:
                        continue
                
                img_info = coco.imgs[a['image_id']]
                img_h, img_w = img_info['height'], img_info['width']
                x, y, w, h = a['bbox']
                xc, yc = x + w / 2, y + h / 2
                xc /= img_w
                yc /= img_h
                w /= img_w
                h /= img_h

                keypoints = np.array(a['keypoints']).reshape([-1, 3])

                # some of crowdpose keypoints are just outside image so clip to image extents
                if not is_coco:
                    pass

                with open(osp.join(labels_path, f"{osp.splitext(img_info['file_name'])[0]}.txt")) as f:
                    s = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(0, xc, yc, w, h)
                    if data['pose_obj']:
                        for i, (x, y, v) in enumerate(keypoints):
                            s += ' {:.6f} {:.6f} {:.6f}'.format(x / img_w, y/img_h, v)
                    s += '\n'
                    f.write(s)

                    # write keypoint objects
                    for i, (x, y, v) in enumerate(keypoints):
                        if v:
                            if isinstance(data['kp_bbox'], list):
                                kp_bbox = data['kp_bbox'][i]
                            else:
                                kp_bbox = data['kp_bbox']
                            kp_bbox *= max(img_w, img_h)
                            
                            s = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(i + 1, x / img_w, y / img_h, kp_bbox / img_w, kp_bbox / img_h)

                            if data['pose_obj']:
                                for _ in range(keypoints.shape[0]):
                                    s += ' {:.6f} {:.6f} {:.6f}'.format(0, 0, 0)
                            s += '\n'
                            f.write(s)
            pbar.close()
        
        with open(img_txt_path, 'w') as f:
            f.write(osp.join(data['path'], 'images', f"{split if is_coco else ''}"), img_info['file_name'] + '\n')


