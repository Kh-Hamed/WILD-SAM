# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import numpy as np
import yaml
import string


# FS_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']
FS_CLASSES = ['Car', 'Pedestrian', 'Bike']


def generate_labels(data_single_frame):
    annotations = {}
    gt_boxes = []
    gt_names = []
    for i in range(len(data_single_frame)):
        box3d = data_single_frame[i]['box3d']
        category = data_single_frame[i]['category']
        gt_boxes.append([box3d['location']['x'],box3d['location']['y'],box3d['location']['z'],
                            box3d['dimension']['length'],box3d['dimension']['width'],box3d['dimension']['height'],
                            box3d['orientation']['z_rotation']])
        gt_names.append(string.capwords(category))

    gt_boxes = np.array(gt_boxes)
    gt_names = np.array(gt_names, dtype=str)
    fs_classes_array = np.array(FS_CLASSES)
    mask = np.isin(gt_names, fs_classes_array)
    annotations['gt_boxes_lidar'] = gt_boxes[mask]
    annotations['gt_names'] = gt_names[mask]
    return annotations



def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False):
    parent_dir = os.path.basename(os.path.dirname(sequence_file))  # Gets 'Batch2'
    file_name = os.path.basename(sequence_file)                   # Gets '1692389221417853952_label3d.yaml'
    sequence_name = os.path.join(parent_dir, file_name)
    # sequence_name = os.path.join(sequence_file.parts[-3], sequence_file.parts[-2], sequence_file.parts[-1][:-13])

    data_labels = yaml.safe_load(open(sequence_file))

    info = {}
    info['label'] = sequence_name
    pc_info = {'num_features': 3, 'lidar_sequence': data_labels['index']}
    # pc_info = {'num_features': 3, 'lidar_sequence': sequence_name}
    info['point_cloud'] = pc_info

    if has_label:
        annotations = generate_labels(data_labels['labels'])
        info['annos'] = annotations

    return info
