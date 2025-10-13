used_classes_fs=['Car', 'Pedestrian', 'Bike']
import copy
import pickle
import glob
import os
import json
import yaml
import string
import numpy as np
# import open3d as o3d
from os import listdir
from os.path import exists,isfile
from skimage import io
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.utils import common_utils, box_utils
from pypcd4 import PointCloud
# from pypcd import *
import torch
import multiprocessing
from pathlib import Path
from functools import partial
from tqdm import tqdm
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

class FourSeasonDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )


        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.split_dir = self.root_path / (self.split + '.txt')
        self.points_list = []
        self.labels_list = [] 
        # self.th_num_pts = [5,5,5]

        self.path_lidar = dataset_cfg.PATH_LIDAR
        self.path_label = dataset_cfg.PATH_LABEL

        self.fs_infos = []
        self.labels_list_sampled = []
        self.set_split()
        self.include_fs_data(self.mode)

        


    def include_fs_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Fourseason dataset')
        fs_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                fs_infos.extend(infos)

        self.fs_infos = fs_infos
        

        if self.logger is not None:
            self.logger.info('Total samples for Fourseason dataset: %d' % (len(self.fs_infos)))
    
    def set_mode(self, mode):
        self.mode = mode

    def set_split(self, split = None):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        if split is not None:
            self.split_dir = self.root_path / (split + '.txt')
        else:
            self.split_dir = self.root_path  / (self.split + '.txt')
        
        if self.split_dir.exists():
            fopen = open(self.split_dir, 'r')
            relative_path = fopen.readlines()
            fopen.close()        

            names_with_Batch = [f[0:-14] for f in relative_path]
            names = [f.split('/')[1] for f in names_with_Batch] 
            self.names_list = names          
            self.points_list = [self.path_lidar+'/'+f+'_oust.pcd' for f in names]
            # self.points_list = [self.path_lidar+'/'+f+'_oust.txt' for f in names]
            self.labels_list = [self.path_label+'/'+f+'_label3d.yaml' for f in names_with_Batch]
        else:
            self.points_list = []
            self.labels_list = []
            self.names_list = []



    def get_lidar(self, sequence_name, Target = False,  path = None):
        if path ==None and not Target:
            
            lidar_file = Path(self.path_lidar) / Path(sequence_name + '_oust.pcd' )
            points_all = PointCloud.from_path(lidar_file)
            pc = points_all.numpy().astype(np.float32)     
            # pc[:, 3] = np.log10(pc[:, 3]  + 1)
            pointcloud = pc[:, 0:4]
        return pointcloud



    def __len__(self):
        # return len(self.points_list)
        if self._merge_all_iters_to_one_epoch:
            return len(self.fs_infos) * self.total_epochs
        return len(self.fs_infos)

    def __getitem__(self, index): 
        
        # index = 520
        
        index = index % len(self.fs_infos)

        info = copy.deepcopy(self.fs_infos[index])

        ######################################################################
        points_T = np.zeros((0, ))
        gt_boxes_T= np.zeros((0, 7))
        gt_names_T = np.zeros((0, 7)).astype(str)
        if index < len(self.infos_T0) and (self.training):
            info_T0 = copy.deepcopy(self.infos_T0[index])
            points_T = self.get_lidar(info_T0['frame_id'], Target= True)

            thresh = 0.60
            taw = 0.20
            msk0 = info_T0['score'] >= thresh
            max_lwh = 1.25 * np.array([4.67, 2.09, 1.71]).reshape(-1, 3)
            lwh0 = info_T0['boxes_lidar'][:, 3:6]
            msk_lwh0 = (lwh0 <= max_lwh).all(axis=1)
            msk0 = msk0 & msk_lwh0
            gt_boxes_T0 = info_T0['boxes_lidar'][msk0]
            gt_boxes_T0_noisy = info_T0['boxes_lidar'][~msk0]
            gt_names_T0 = info_T0['name'][msk0]

            # gt_boxes_T = gt_boxes_T0
            # gt_boxes_T_noisy = gt_boxes_T0_noisy
            # gt_names_T = gt_names_T0
            
            ##########################################################################################
            ##########################################################################################

            info_T1 = copy.deepcopy(self.infos_T1[index])
            msk1 = info_T1['score'] >= thresh
            lwh1 = info_T1['boxes_lidar'][:, 3:6]
            msk_lwh1 = (lwh1 <= max_lwh).all(axis=1)
            msk1 = msk1 & msk_lwh1
            gt_boxes_T1 = info_T1['boxes_lidar'][msk1]
            gt_names_T1 = info_T1['name'][msk1]
            gt_scores_T1 = info_T1['score'][msk1]
        
            
            gt_boxes_T1_noisy = info_T1['boxes_lidar'][~msk1]
            from pcdet.ops.iou3d_nms import iou3d_nms_utils
            if gt_boxes_T0.shape[0] != 0 and gt_boxes_T1.shape[0] != 0:
                iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(gt_boxes_T1[:, 0:7], gt_boxes_T0[:, 0:7])

                best_T1_to_T0 = iou_matrix.argmax(axis=1)  # (N1,) best T0 index for each T1
                best_T0_to_T1 = iou_matrix.argmax(axis=0)  # (N0,) best T1 index for each T0

                matched_T0_indices = []
                matched_T1_indices = []

                for i1, i0 in enumerate(best_T1_to_T0):
                    if best_T0_to_T1[i0] == i1 and iou_matrix[i1, i0] > 0.7:
                        matched_T0_indices.append(i0)
                        matched_T1_indices.append(i1)
                matched_T0 = np.array(matched_T0_indices, dtype=np.int32)
                matched_T1 = np.array(matched_T1_indices, dtype=np.int32)
                unmatched_T0 = np.setdiff1d(np.arange(gt_boxes_T0.shape[0]), matched_T0).astype(np.int32)
                unmatched_T1 = np.setdiff1d(np.arange(gt_boxes_T1.shape[0]), matched_T1).astype(np.int32)

            elif gt_boxes_T0.shape[0] == 0 and gt_boxes_T1.shape[0] != 0:
                matched_T0 = np.zeros((0,), dtype=np.int32)
                unmatched_T0 = np.zeros((0,), dtype=np.int32)
                matched_T1 = np.zeros((0,), dtype=np.int32)
                unmatched_T1 = np.setdiff1d(np.arange(gt_boxes_T1.shape[0]), matched_T1)
            elif gt_boxes_T0.shape[0] != 0 and gt_boxes_T1.shape[0] == 0:
                matched_T0 = np.zeros((0,), dtype=np.int32)
                unmatched_T0 = np.setdiff1d(np.arange(gt_boxes_T0.shape[0]), matched_T0)
                matched_T1 = np.zeros((0,), dtype=np.int32)
                unmatched_T1 = np.zeros((0,), dtype=np.int32)
            else:
                matched_T0 = np.zeros((0,), dtype=np.int32)
                unmatched_T0 = np.zeros((0,), dtype=np.int32)
                matched_T1 = np.zeros((0,), dtype=np.int32)
                unmatched_T1 = np.zeros((0,), dtype=np.int32)


            matched_boxes_T1 = gt_boxes_T1[matched_T1]
            matched_names_T1 = gt_names_T1[matched_T1]

            new_gt_names_T1 =  gt_names_T1[unmatched_T1]
            new_gt_boxes_T1 =  gt_boxes_T1[unmatched_T1]

            new_scores_T1 = gt_scores_T1[unmatched_T1]
            msk_new1 = new_scores_T1 >= (thresh + taw)

            new_reliable_boxes_T1 = new_gt_boxes_T1[msk_new1]
            new_reliable_names_T1 = new_gt_names_T1[msk_new1]

            new_unreliable_gt_boxes_T1 = new_gt_boxes_T1[~msk_new1]
            unmatched_boxes_T0 = gt_boxes_T0[unmatched_T0]

            gt_boxes_T = np.concatenate((matched_boxes_T1, new_reliable_boxes_T1), axis=0)
            gt_boxes_T_noisy = np.concatenate((gt_boxes_T1_noisy, gt_boxes_T0_noisy, unmatched_boxes_T0, new_unreliable_gt_boxes_T1), axis=0)
            gt_names_T = np.concatenate((matched_names_T1, new_reliable_names_T1), axis=0)

            ################################################################################################
            ################################################################################################
            
            gt_boxes_T_noisy = box_utils.enlarge_box3d(
            gt_boxes_T_noisy[:, 0:7], extra_width=(0.25, 0.25, 0.0)
            )   
            # gt_names_T = info_T['annos']['name']
            # gt_boxes_T = info_T['annos']['gt_boxes_lidar'][:, 0:7]
            point_masks_noisy_boxes = roiaware_pool3d_utils.points_in_boxes_cpu(points_T[:, 0:3], gt_boxes_T_noisy.numpy())
            point_masks_accurate_boxes = roiaware_pool3d_utils.points_in_boxes_cpu(points_T[:, 0:3], gt_boxes_T)
            msk_bg = point_masks_noisy_boxes.sum(axis=0) == 0
            msk_fg = point_masks_accurate_boxes.sum(axis=0) != 0
            mask_keep = msk_bg | msk_fg
            points_T = points_T[mask_keep]
            # points_T = box_utils.remove_points_in_boxes3d(points_T, gt_boxes_T_noisy)
            input_dict_T = {
            'points': points_T,
            'gt_boxes':gt_boxes_T,
            'gt_names':gt_names_T,
            'frame_id': info_T0['frame_id'],
            'calib': None,
            'image_shape': 0
        }

        ######################################################################

        points = self.get_lidar(info['point_cloud']['lidar_sequence'])

        
        gt_boxes = info['annos']['gt_boxes_lidar']
        gt_names = info['annos']['gt_names']
        input_dict = {
            'points': points,
            'gt_boxes':gt_boxes,
            'gt_names':gt_names,
            'frame_id': info['point_cloud']['lidar_sequence'],
            'calib': None,
            'image_shape': 0
        }

        input_dict_modulated = copy.deepcopy(input_dict)
        ###################################################################################
        if self.training:
            input_dict_modulated['src_modulated'] = True
            data_dict_src_m = self.prepare_data(data_dict=input_dict_modulated)
            data_dict_src_m['metadata'] = info.get('metadata', info['frame_id'])
            data_dict_src_m.pop('num_points_in_gt', None)
            data_dict_src_m.pop('src_modulated', None)
            # return [data_dict, data_dict_src_m]
            if points_T.shape[0] != 0:
                data_dict_T = self.prepare_data(data_dict=input_dict_T, Target= True)
                data_dict_T['metadata'] = info_T0['metadata']
                return [data_dict, data_dict_src_m , data_dict_T]
                # return [data_dict , data_dict_T]
            else:
                return [data_dict, data_dict_src_m]
                # return [data_dict]

        else:
            return [data_dict]
        ###################################################################################

    
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'Car': 'Car',
            'Pedestrian': 'Pedestrian',
            'Bike': 'Cyclist',
        }
        class_names_new = ['Car', 'Pedestrian', 'Cyclist']


        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            #for anno in annos:
            for ii in range(len(annos)):
                anno = annos[ii]
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                anno['name'] = anno['name'].tolist()
                for k in range(len(anno['name'])):
                    if anno['name'][k] in map_name_to_kitti:
                        name = anno['name'][k]
                        name_new = map_name_to_kitti[name]
                        anno['name'][k] = name_new
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes_lidar'].copy()
                                                     

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, is_gt=False)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.fs_infos]

        return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)

    # def evaluation(self, det_annos, gt_annos, class_names, **kwargs):
    #     eval_det_annos = copy.deepcopy(det_annos)
    #     eval_gt_annos = copy.deepcopy(gt_annos)
    #     return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
    


    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, update_info_only=False):
        from pcdet.datasets.fourseason import fourseason_utils
        print('---------------The fourseason sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.labels_list)))

        process_single_sequence = partial(
            fourseason_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, update_info_only=update_info_only
        )
        labels_list_sampled = self.labels_list[::sampled_interval]
        labels_list_sampled = [Path(path) for path in labels_list_sampled]
        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, labels_list_sampled),
                                       total=len(labels_list_sampled)))
        # sequence_infos = process_single_sequence(labels_list_sampled[23])
        for info in tqdm(sequence_infos, desc="Processing sequences"):
            points = self.get_lidar(info['point_cloud']['lidar_sequence'])

            names = info['annos']['gt_names']
            gt_boxes = info['annos']['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue


            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
            gt_pts = []
            for i in range(num_obj):
                gt_points = points[box_idxs_of_pts == i]
                if gt_points.shape[0]>= 5 :
                    gt_pts.append(True)
                else:
                    gt_pts.append(False)
            gt_pts = np.array(gt_pts)
            info['annos']['gt_names'] = names[gt_pts]
            info['annos']['gt_boxes_lidar'] = gt_boxes[gt_pts]
        return sequence_infos


    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):

    

        database_save_path = save_path / ('%s_LISA_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_LISA_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        # db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        for k in tqdm(range(0, len(infos), sampled_interval)):
            info = infos[k]
            sequence_name = info['point_cloud']['lidar_sequence']
            # sequence_name = pc_info['lidar_sequence']
            points = self.get_lidar(sequence_name)
            names = info['annos']['gt_names']
            gt_boxes = info['annos']['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue


            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sequence_name, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                if gt_points.shape[0]<5:
                    continue
                gt_points[:, 0:3] -= gt_boxes[i, 0:3]

                if (used_classes is None) or names[i] in used_classes:
                    gt_points = gt_points.astype(np.float32)
                    assert gt_points.dtype == np.float32
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                                'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0]}


                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    


def create_fs_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='fs',
                       workers=min(16, multiprocessing.cpu_count()), update_info_only=False):
    dataset = FourSeasonDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    # val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    fs_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, update_info_only=update_info_only
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(fs_infos_train, f)
    print('----------------fourseason info train file is saved to %s----------------' % train_filename)

    # dataset.set_split(val_split)
    # fs_infos_val = dataset.get_infos(
    #     raw_data_path=data_path / raw_data_tag,
    #     save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #     sampled_interval=1, update_info_only=update_info_only
    # )
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(fs_infos_val, f)
    # print('----------------fourseason info val file is saved to %s----------------' % val_filename)

    if update_info_only:
        returns

    print('---------------Start create groundtruth database for data augmentation---------------')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=used_classes_fs, processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')


def filter_fs_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='fs',
                       th_numpts_list=[5,5,5], info_path_unfiltered=''):
    dataset = FourSeasonDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    dataset.dataset_cfg.INFO_PATH['train'] = [info_path_unfiltered[0]]
    dataset.dataset_cfg.INFO_PATH['test'] = [info_path_unfiltered[1]]
    
    # update training
    dataset.set_split(train_split)
    dataset.include_fs_data('train')
    count_before_filter = 0
    count_after_filter = 0
    for index in range(len(dataset.fs_infos)):
        info = copy.deepcopy(dataset.fs_infos[index])
        points = dataset.get_lidar(info['point_cloud']['lidar_sequence'])
        gt_boxes = info['annos']['gt_boxes_lidar']
        gt_names = info['annos']['gt_names']
        if not gt_boxes.shape[0]:
            continue

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)

        mask_gts = np.zeros(num_obj)
        for i in range(num_obj):
            name = gt_names[i]
            label = class_names.index(name)
            th_numpts = th_numpts_list[label]
            numpts = np.sum(point_indices[i] > 0)
            if numpts > th_numpts:
                mask_gts[i] = 1.0
        count_before_filter += info['annos']['gt_boxes_lidar'].shape[0]
        info['annos']['gt_boxes_lidar'] = gt_boxes[mask_gts>0.0]
        info['annos']['gt_names'] = gt_names[mask_gts>0.0]
        count_after_filter += info['annos']['gt_boxes_lidar'].shape[0]

    with open(train_filename, 'wb') as f:
        pickle.dump(dataset.fs_infos, f)
    print('train count_before_filter %s' %str(count_before_filter))
    print('train count_after_filter %s' %str(count_after_filter))
    print('----------------fourseason filtered info train file is saved to %s----------------' % train_filename)

    # update validation
    count_before_filter = 0
    count_after_filter = 0    
    dataset.set_split(val_split)
    dataset.include_fs_data('test')
    for index in range(len(dataset.fs_infos)):
        info = copy.deepcopy(dataset.fs_infos[index])
        points = dataset.get_lidar(info['point_cloud']['lidar_sequence'])
        gt_boxes = info['annos']['gt_boxes_lidar']
        gt_names = info['annos']['gt_names']
        if not gt_boxes.shape[0]:
            continue

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)

        mask_gts = np.zeros(num_obj)
        for i in range(num_obj):
            name = gt_names[i]
            label = class_names.index(name)
            th_numpts = th_numpts_list[label]
            numpts = np.sum(point_indices[i] > 0)
            if numpts > th_numpts:
                mask_gts[i] = 1.0
        count_before_filter += info['annos']['gt_boxes_lidar'].shape[0]
        info['annos']['gt_boxes_lidar'] = gt_boxes[mask_gts>0.0]
        info['annos']['gt_names'] = gt_names[mask_gts>0.0]
        count_after_filter += info['annos']['gt_boxes_lidar'].shape[0]


    with open(val_filename, 'wb') as f:
        pickle.dump(dataset.fs_infos, f)
    print('val count_before_filter %s' %str(count_before_filter))
    print('val count_after_filter %s' %str(count_after_filter))
    print('----------------fourseason filtered info val file is saved to %s----------------' % train_filename)


def create_fs_gt_database(
    dataset_cfg, class_names, data_path, save_path, processed_data_tag='fs',
    workers=min(16, multiprocessing.cpu_count()), use_parallel=False, crop_gt_with_tail=False):
    dataset = FourSeasonDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    train_split = 'train'
    train_filename = '/space/userfiles/khatouna/SAM_IV_conference/data/LISA/snow_10_mm_hr/fs_infos_train.pkl'

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)

    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=used_classes_fs, processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/space/userfiles/khatouna/SAM_IV_conference/tools/cfgs/dataset_configs/fourseason_dataset_LISA.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_fs_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='fs', help='')
    parser.add_argument('--update_info_only', action='store_true', default=False, help='')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='')
    parser.add_argument('--wo_crop_gt_with_tail', action='store_true', default=False, help='')
    parser.add_argument('--data_folder', type=str, default='', help='specify the folder of dataset')

    args = parser.parse_args()

    # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

    # ROOT_DIR = Path('data/fourseason/ImageSets/2022_rain_5min_balanced')
    ROOT_DIR = Path('/space/userfiles/khatouna/SAM_IV_conference/data/LISA/snow_10_mm_hr')
    # ROOT_DIR = Path('data/fourseason/ImageSets/')

    if args.func == 'filter_fs_infos':
        ROOT_DIR = Path(args.data_folder)
        info_path_unfiltered = ['fs_infos_train_unfiltered.pkl', 'fs_infos_val_unfiltered.pkl']

    if args.func == 'create_fs_infos':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_fs_infos(
            dataset_cfg=dataset_cfg,
            class_names=used_classes_fs,
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            raw_data_tag='',
            processed_data_tag=args.processed_data_tag,
            update_info_only=args.update_info_only
        )
    elif args.func == 'create_fs_gt_database':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_fs_gt_database(
            dataset_cfg=dataset_cfg,
            class_names=used_classes_fs,
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            processed_data_tag=args.processed_data_tag,
            use_parallel=args.use_parallel, 
            crop_gt_with_tail=not args.wo_crop_gt_with_tail
        )
    else:
        raise NotImplementedError
