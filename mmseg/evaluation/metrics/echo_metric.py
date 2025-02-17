from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import os.path as osp

from PIL import Image

from mmseg.registry import METRICS
from mmengine.logging import MMLogger, print_log

from .iou_metric import IoUMetric
from scipy.spatial.distance import cdist
from echo.tools.utils_contour import find_contour_points
from echo.tools.ef_simpson_ljy import get_volume
from echo.tools.compute_ef import compute_left_ventricle_volumes_single_plane
from medpy.metric.binary import hd95 as medpy_hd95

def draw_linear_regression_map(data, xname:str, yname:str, fig_name:str):
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    # tips = sns.load_dataset("tips")
    g = sns.jointplot(x=xname, y=yname, data=data,
                    kind="reg", truncate=False,
                    xlim=(0, 100), ylim=(0, 100),
                    color="m", height=7)
    g.savefig(fig_name+'.png')
    return g


def hausdorff_distance(mask1: Tensor, mask2: Tensor, percentile: int = 95):
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
    assert percentile >= 0 and percentile <= 100, 'percentile invaild'
    hausdorff_dist = np.percentile(dist, percentile)

    return hausdorff_dist

def corr(x, y):
    '''
        x : gt
        y : pred
        A = mean( (y_real - mean(y_real)) * (y_predict - mean(y_predict)) )
        B = std(y_real) * std(y_predict)
        corr = A / B
    '''
    A = ((x - x.mean()) * (y - y.mean())).mean()
    B = std(x) * std(y)
    corr = A / B
    return corr

def bias(x, y):
    '''
        x : gt
        y : pred
        bias = sum( abs(y_real - y_predict) ) / len( y_real )
    '''
    # return ((x - y)).mean()
    return (abs(x - y)).mean()

def std(x):
    '''
        A = (y - mean(y)) * (y - mean(y))
        std = sqrt( sum(A) / n )
    '''
    # A = ((x - x.mean()) * (x - x.mean())).mean()
    # std = np.sqrt(A)
    return np.std(x)

def simpson(label, pred_label, gt_vol):
    pred_vol = pred_label.sum() * gt_vol / label.sum()
    return pred_vol

def simpson_rule(pred_label, spacing = None):
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.cpu().data.numpy().astype(np.uint8)
    elif isinstance(pred_label, np.ndarray):
        pred_label = pred_label.astype(np.uint8)

    if spacing is not None:
        pred_vol = get_volume(pred_label, spacing=spacing)
    else:
        pred_vol = get_volume(pred_label)
    return torch.tensor(pred_vol)

def simpson_single_plane(pred_label, spacing = None):
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.cpu().data.numpy().astype(np.uint8)
    elif isinstance(pred_label, np.ndarray):
        pred_label = pred_label.astype(np.uint8)


def LVEF(vol_pred):
    '''
        pixel to volume
    '''
    vol1, vol2 = vol_pred
    a = abs(vol1 - vol2)
    b = max(vol1, vol2)
    # a = vol2 - vol1 
    # b = vol2
    return a / b *100

@METRICS.register_module()
class EchoMetric(IoUMetric):

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ...,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(ignore_index, iou_metrics, nan_to_num, beta,
                         collect_device, output_dir, format_only, prefix,
                         **kwargs)
        self.hd95 = []
        self.lvef_gt = []
        self.lvef_pred = []
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if isinstance(self.lvef_gt[0], Tensor):
            self.lvef_gt = torch.stack(self.lvef_gt).numpy()
        if isinstance(self.lvef_pred[0], Tensor):
            self.lvef_pred = torch.stack(self.lvef_pred).numpy()
        if isinstance(self.lvef_pred, List):
            self.lvef_pred = np.array(self.lvef_pred)
        self.corr = corr(self.lvef_gt, self.lvef_pred)
        self.bias = bias(self.lvef_gt, self.lvef_pred)
        self.std = std(self.lvef_pred-self.lvef_gt)
        self.hd95 =np.mean(self.hd95)
        print_log('corr: ' + str((self.corr*100).round(2)), logger=logger)
        print_log('bias: ' + str((self.bias).round(2)), logger=logger)
        print_log('std : ' + str((self.std).round(2)), logger=logger)
        print_log('HD95: ' + str((self.hd95).round(2)), logger=logger)

        dataframe1 = pd.DataFrame(self.lvef_gt,columns=['gt'])
        dataframe2 = pd.DataFrame(self.lvef_pred,columns=['pred'])
        df = pd.concat([dataframe1,dataframe2], axis=1)

        draw_linear_regression_map(df, xname='gt', yname='pred',fig_name='1')
        return super().compute_metrics(results)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        assert len(data_samples) == 2
        esv, edv = data_samples[0]['esv'], data_samples[0]['edv']
        vol_gt = [esv,edv]
        vol_pred = []
        for idx, data_sample in enumerate(data_samples):
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                # compute hd95
                # hd95 = hausdorff_distance(pred_label, label, 95)
                # self.hd95.append(hd95)
                # compute by medpy_hd95
                med_hd95 = medpy_hd95(
                    pred_label.cpu().data.numpy(),
                    label.cpu().data.numpy(),
                    voxelspacing=data_sample["spacing"][:-1],
                )
                self.hd95.append(med_hd95)
                
                # compute vol_pred
                # vol_pred.append(simpson(label,pred_label,vol_gt[idx]))
                vol_pred.append(
                    simpson_rule(
                        label,
                        spacing=data_sample["spacing"][:-1]
                    )
                )
                # vol_pred.append(
                #     compute_left_ventricle_volumes_single_plane(
                #         pred_label.cpu().data.numpy(),
                #         voxelspacing=data_sample["spacing"].cpu().data.numpy()[:-1]
                #     )
                # )

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

        # lvef
        self.lvef_gt.append(data_samples[0]['ef'])
        self.lvef_pred.append(LVEF(vol_pred))