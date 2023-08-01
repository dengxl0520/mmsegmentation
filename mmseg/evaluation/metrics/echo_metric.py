from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
import os.path as osp

from PIL import Image

from mmseg.registry import METRICS

from .iou_metric import IoUMetric
from scipy.spatial.distance import cdist
from echo.tools.utils_contour import find_contour_points


def hausdorff_distance(mask1: Tensor, mask2: Tensor, percentile: int=95):
    # 获取目标集合和预测集合中的非零点的坐标
    mask1 = mask1.cpu()
    mask2 = mask2.cpu()

    contours1 = find_contour_points(mask1)
    contours2 = find_contour_points(mask2)
    if contours1.size == 0 or contours2.size == 0:
        return 0

    # 计算所有点对之间的欧氏距离
    dist = cdist(contours1, contours2)
    dist = np.concatenate((np.min(dist, axis=0), np.min(dist, axis=1)))
    assert percentile >= 0 and percentile <=100, 'percentile invaild'
    hausdorff_dist = np.percentile(dist, percentile)

    return hausdorff_dist

def coor(x, y):
    '''
    A = mean( (y_real - mean(y_real)) * (y_predict - mean(y_predict)) )
    B = std(y_real) * std(y_predict)
    corr = A / B
    ''' 

def bias(x, y):
    '''
    bias = sum( y_real - y_predict ) / len( y_real )
    '''


def std(x, y):
    '''
    A = (y - mean(y)) * (y - mean(y))
    std = sqrt( sum(  ) / n )
    '''
    
def LVEF(x, y):
    a = abs(x-y)
    b = max(x, y)
    return a / b
    

@METRICS.register_module()
class EchoMetric(IoUMetric):
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                # compute hd95
                hd95 = hausdorff_distance(pred_label, label, 95)
                print(hd95)
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)
