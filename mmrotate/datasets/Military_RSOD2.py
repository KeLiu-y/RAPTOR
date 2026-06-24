# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class MilitaryRSODDataset2(CustomDataset):
    """Military-RSOD dataset for detection."""

    # 更新后的 53 个类别
    CLASSES = ('ABD', 'AFV', 'Airport', 'Auxiliary', 'B-1B', 'B-52', 'BF',
               'Bridge', 'Business Jet', 'C-130', 'C-17', 'C-5', 'CYSD',
               'DD', 'DDG-1000', 'Dam', 'E-3', 'E-8', 'F-15', 'F-16',
               'F-22', 'F-35', 'FA-18', 'GGD', 'Helicopter', 'Helipad',
               'INSV', 'Inf', 'KC-10', 'KC-135', 'KHAA', 'LCS', 'LHD',
               'LMV', 'MCV', 'NAA', 'Overpass', 'P-3C', 'PC',
               'Propeller_Aircraft', 'SAATD', 'SGD', 'SMV', 'SU-24',
               'SU-34', 'SU-35', 'Submarine', 'TC', 'TU-160', 'TU-22',
               'TU-95', 'Trainer_Aircraft', 'WLHD')

    # 更新后的调色板 (53种颜色，对应53个类别)
    PALETTE = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
        (92, 0, 73), (133, 129, 255), (78, 180, 149), (0, 228, 0), (174, 255, 243),
        (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255),
        (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61),
        (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
        (0, 143, 149), (179, 0, 194), (209, 99, 106)
    ]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(MilitaryRSODDataset2, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
        Args:
            ann_folder: folder that contains DOTA v1 annotations txt files
        """
        # 定义所有允许的图片后缀（小写），用于扫描过滤
        valid_suffixes = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        ann_files = glob.glob(ann_folder + '/*.txt')
        
        # === 步骤1: 智能推断图片文件夹路径 ===
        # DOTA 数据集通常结构: train/labelTxt 和 train/images
        if 'labelTxt' in ann_folder:
            img_folder = ann_folder.replace('labelTxt', 'images')
        elif 'annfiles' in ann_folder:
            img_folder = ann_folder.replace('annfiles', 'images')
        else:
            # 默认尝试同级的 images 文件夹
            img_folder = osp.join(osp.dirname(ann_folder), 'images')

        # === 步骤2: 扫描图片文件夹，建立 {无后缀ID: 带后缀文件名} 的映射表 ===
        img_id_map = {}
        if os.path.exists(img_folder):
            print(f'Scanning image folder: {img_folder} ...')
            for filename in os.listdir(img_folder):
                name, ext = os.path.splitext(filename)
                # 只要后缀在允许列表中（忽略大小写），就加入映射
                if ext.lower() in valid_suffixes:
                    img_id_map[name] = filename
        else:
            print(f'[Warning] Image folder not found at: {img_folder}')

        data_infos = []
        
        # === 测试阶段（如果没找到txt）===
        if not ann_files: 
            # 直接使用扫描到的图片列表
            for img_name in img_id_map.values():
                data_info = {}
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        
        # === 训练/验证阶段 ===
        else:
            for ann_file in ann_files:
                data_info = {}
                # 获取 txt 文件名作为 ID (例如 'Military_001.txt' -> 'Military_001')
                img_id = osp.split(ann_file)[1][:-4]
                
                # === 步骤3: 从映射表中查找对应的图片文件名 ===
                if img_id in img_id_map:
                    img_name = img_id_map[img_id]
                else:
                    # 如果 txt 存在但找不到对应的图片，跳过该样本
                    # print(f'[Warning] Image not found for {img_id}, skipping.')
                    continue
                # ==========================================

                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        # 格式检查
                        if len(bbox_info) < 9:
                            continue

                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:
                            continue
                        
                        # 处理 difficulty 和 带空格的类别名
                        try:
                            difficulty = int(bbox_info[-1])
                            cls_name = ' '.join(bbox_info[8:-1])
                        except ValueError:
                            # 容错：如果没有 difficulty 列
                            difficulty = 0
                            cls_name = ' '.join(bbox_info[8:])
                            
                        if cls_name not in cls_map:
                            continue
                            
                        label = cls_map[cls_name]
                        
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros((0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros((0, 8), dtype=np.float32)

                data_infos.append(data_info)

        print(f'Loaded {len(data_infos)} images from {ann_folder}')
        self.img_ids = [*map(lambda x: x['filename'].rsplit('.', 1)[0], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.

        Returns:
            list: merged results.
        """

        def extract_xy(img_id):
            """Extract x and y coordinates from image ID.

            Args:
                img_id (str): ID of the image.

            Returns:
                Tuple of two integers, the x and y coordinates.
            """
            pattern = re.compile(r'__(\d+)___(\d+)')
            match = pattern.search(img_id)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return x, y
            else:
                warnings.warn(
                    "Can't find coordinates in filename, "
                    'the coordinates will be set to (0,0) by default.',
                    category=Warning)
                return 0, 0

        collector = defaultdict(list)
        for idx, img_id in enumerate(self.img_ids):
            result = results[idx]
            oriname = img_id.split('__', maxsplit=1)[0]
            x, y = extract_xy(img_id)
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))
            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Executing on Single Processor')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print(f'Executing on {nproc} processors')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        # Return a zipped list of merged results
        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                                iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results