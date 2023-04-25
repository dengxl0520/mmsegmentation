from typing import Optional, Dict
import numpy as np

import mmcv

from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms import to_tensor

from mmseg.structures.seg_data_sample import PixelData
from mmseg.structures.seg_video_data_sample import SegEchoDataSample

from mmcv.transforms.processing import RandomFlip, RandomResize
from .transforms import GenerateEdge, RandomCrop, PhotoMetricDistortion


def load_video_and_mask_file(img_path: str, anno_path: str, frame_length: int = 10):
    '''
        modify by ultrasound_npy_sequence_load() from EchoGraphs
        args:
            img_path: videos .npy file path
            anno_path: annotations .npz file path
            frame_length: Placing ED frame at index 0, and ES frame at index -1
        return: 
            imgs: (F,3,112,112)
            masks: (2,112,112)  ED and ES frame
            ef: float
    '''
    # load video 
    video = np.load(img_path, allow_pickle=True)
    video = video.swapaxes(0, 1)
    kpts_list = np.load(anno_path, allow_pickle=True)
    ef, vol1, vol2 = kpts_list['ef'], kpts_list['vol1'], kpts_list['vol2']
    
    # Collect masks:
    idx_list = []
    mask_list = kpts_list['fnum_mask'].tolist()
    masks = []
    for kpt in kpts_list['fnum_mask'].tolist().keys():
        idx_list.append(int(kpt))
        masks.append(mask_list[kpt])
       
    # Swap if ED before ES:
    if idx_list[0] > idx_list[1]:
        idx_list.reverse()
        masks.reverse()
        vol1, vol2 = vol2, vol1

    # compute step:
    x0, x1 = idx_list[1], idx_list[0]
    step = min(x0, (x0 - x1) / (frame_length - 1))
    
    # select frames inds:
    frame_inds = [int(idx_list[0] + step * i) for i in range(frame_length)]

    # Collect frames:
    frames = []
    for i in range(frame_length):
        frames.append(video[frame_inds[i]])

    masks = np.asarray(masks)
    imgs = np.asarray(frames)

    return imgs, masks, ef


@TRANSFORMS.register_module()
class LoadVideoAndAnnoFromFile(BaseTransform):
    '''Load n frames video from .npy file
        args:
            frame_length: (int) least 2
    '''

    def __init__(self, frame_length = 10) -> None:
        self.frame_length = frame_length

    def transform(self, results: dict) -> Optional[dict]:
        img_path = results['img_path']
        anno_path = results['seg_map_path']
        imgs, masks, ef = load_video_and_mask_file(img_path, anno_path, self.frame_length)

        results['imgs'] = imgs
        results['ori_imgs'] = imgs
        results['masks'] = masks
        results['ef'] = ef
        results['imgs_shape'] = imgs.shape
        results['img_shape'] = imgs.shape[2:]
        results['ori_shape'] = imgs.shape[2:]

        return results

@TRANSFORMS.register_module()
class VideoGenerateEdge(GenerateEdge):
    def transform(self, results: Dict) -> Dict:
        mask_edge = []
        for i in range(len(results["masks"])):
            results['gt_seg_map'] = results['masks'][i]
            super().transform(results)
            mask_edge.append(results['gt_edge_map'])
        results['mask_edge'] = mask_edge
        return results


@TRANSFORMS.register_module()
class VideoRandomResize(RandomResize):
    def transform(self, results: dict) -> dict:
        return super().transform(results)
    
@TRANSFORMS.register_module()
class VideoRandomFlip(RandomFlip):
    def transform(self, results: dict) -> dict:
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir

            # flip imgs
            results['imgs'] = mmcv.imflip(img=results['imgs'], direction=cur_dir)

            # filp masks
            results['masks'] = mmcv.imflip(img=results['masks'], direction=cur_dir)
        return results


@TRANSFORMS.register_module()
class VideoRandomCrop(RandomCrop):
    def transform(self, results: dict) -> dict:
        return super().transform(results)

@TRANSFORMS.register_module()
class VideoPhotoMetricDistortion(PhotoMetricDistortion):
    def transform(self, results: dict) -> dict:
        results['img'] = results['imgs'][0]
        results = super().transform(results)
        return results


@TRANSFORMS.register_module()
class PackSegVideoInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'imgs' in results:
            imgs = results['imgs'].copy()
            imgs = to_tensor(imgs)
            packed_results['inputs'] = imgs

        data_sample = SegEchoDataSample()
        if 'masks' in results:
            frame1 = to_tensor(results['masks'][0,:,:].astype(np.int64))
            frame1.unsqueeze_(0)
            frame2 = to_tensor(results['masks'][1,:,:].astype(np.int64))
            frame2.unsqueeze_(0)
            # use first frame and last frame
            ori_frame1 = results['ori_imgs'][0,:,:,:]
            ori_frame2 = results['ori_imgs'][-1,:,:,:]
            gt_sem_seg_data = dict(frame1=frame1, frame2=frame2, ori_frame1=ori_frame1, ori_frame2=ori_frame2)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        if 'mask_edge' in results:
            frame1_edge = to_tensor(results['mask_edge'][0].astype(np.int64))
            frame1_edge.unsqueeze_(0)
            frame2_edge = to_tensor(results['mask_edge'][1].astype(np.int64))
            frame2_edge.unsqueeze_(0)
            gt_edge_data = dict(frame1_edge=frame1_edge,
                frame2_edge=frame2_edge)
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

                
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        
        img_meta['ef'] = to_tensor(results['ef'].astype(np.float32))
        img_meta['frameidx'] = '0'
                
        data_sample.set_metainfo(img_meta)
    
        packed_results['data_samples'] = data_sample
        return packed_results