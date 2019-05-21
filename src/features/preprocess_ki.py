"""Preprocess annnotations with KITTI ground-truth"""

import os
import glob
import math
import logging
from collections import defaultdict
import json
import datetime
from utils.kitti import get_calibration, check_conditions, split_training
from utils.pifpaf import get_input_data, preprocess_pif
from utils.misc import get_idx_max, append_cluster


class PreprocessKitti:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
              'test': dict(X=[], Y=[], names=[], kps=[], K=[],
                           clst=defaultdict(lambda: defaultdict(list)))}
    dic_names = defaultdict(lambda: defaultdict(list))

    def __init__(self, dir_ann, iou_thresh=0.3):

        self.dir_ann = dir_ann
        self.iou_thresh = iou_thresh
        self.dir_gt = os.path.join('data', 'kitti', 'gt')
        self.names_gt = os.listdir(self.dir_gt)
        self.dir_kk = os.path.join('data', 'kitti', 'calib')
        self.list_gt = glob.glob(self.dir_gt + '/*.txt')
        assert os.path.exists(self.dir_gt), "Ground truth dir does not exist"
        assert os.path.exists(self.dir_ann), "Annotation dir does not exist"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        dir_out = os.path.join('data', 'arrays')
        self.path_joints = os.path.join(dir_out, 'joints-kitti-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-kitti-' + now_time + '.json')
        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        self.set_train, self.set_val = split_training(self.names_gt, path_train, path_val)

    def run(self):

        cnt_gt = 0
        cnt_fnf = 0
        dic_cnt = {'train': 0, 'val': 0, 'test': 0}

        for name in self.names_gt:
            # Extract ground truth
            path_gt = os.path.join(self.dir_gt, name)
            basename, _ = os.path.splitext(name)
            boxes_gt = []
            dds = []

            if name in self.set_train:
                phase = 'train'
            elif name in self.set_val:
                phase = 'val'
            else:
                cnt_fnf += 1
                continue

            # Extract keypoints
            path_txt = os.path.join(self.dir_kk, basename + '.txt')
            kk, tt = get_calibration(path_txt)

            # Iterate over each line of the gt file and save box location and distances
            with open(path_gt, "r") as f_gt:
                for line_gt in f_gt:
                    if check_conditions(line_gt, mode='gt'):
                        box = [float(x) for x in line_gt.split()[4:8]]
                        boxes_gt.append(box)
                        loc_gt = [float(x) for x in line_gt.split()[11:14]]
                        dd = math.sqrt(loc_gt[0] ** 2 + loc_gt[1] ** 2 + loc_gt[2] ** 2)
                        dds.append(dd)
                        self.dic_names[basename + '.png']['boxes'].append(box)
                        self.dic_names[basename + '.png']['dds'].append(dd)
                        self.dic_names[basename + '.png']['K'] = kk.tolist()
                        cnt_gt += 1

            # Find the annotations if exists
            try:
                with open(os.path.join(self.dir_ann, basename + '.png.pifpaf.json'), 'r') as f:
                    annotations = json.load(f)
                boxes, keypoints = preprocess_pif(annotations)
                (inputs, _), (uv_kps, uv_boxes, _, _) = get_input_data(boxes, keypoints, kk)

            except FileNotFoundError:
                uv_boxes = []

            # Match each set of keypoint with a ground truth
            for ii, box in enumerate(uv_boxes):
                idx_max, iou_max = get_idx_max(box, boxes_gt)

                if iou_max >= self.iou_thresh:

                    self.dic_jo[phase]['kps'].append(uv_kps[ii])
                    self.dic_jo[phase]['X'].append(inputs[ii])
                    self.dic_jo[phase]['Y'].append([dds[idx_max]])  # Trick to make it (nn,1)
                    self.dic_jo[phase]['K'] = kk.tolist()
                    self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                    append_cluster(self.dic_jo, phase, inputs[ii], dds[idx_max], uv_kps[ii])
                    dic_cnt[phase] += 1
                    boxes_gt.pop(idx_max)
                    dds.pop(idx_max)

        with open(self.path_joints, 'w') as file:
            json.dump(self.dic_jo, file)
        with open(os.path.join(self.path_names), 'w') as file:
            json.dump(self.dic_names, file)
        for phase in ['train', 'val', 'test']:
            print("Saved {} annotations for phase {}"
                  .format(dic_cnt[phase], phase))
        print("Number of GT files: {}. Files not found: {}"
              .format(cnt_gt, cnt_fnf))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))



