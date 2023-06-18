import numpy as np
from numpy import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmcv

from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms import to_tensor

from mmseg.structures.seg_data_sample import PixelData
from mmseg.structures.seg_video_data_sample import SegEchoDataSample

from mmcv.image.geometric import _scale_size
from mmcv.transforms.processing import RandomFlip, RandomResize, Resize
from .transforms import GenerateEdge, RandomCrop, PhotoMetricDistortion


def load_video_and_mask_file(img_path: str,
                             anno_path: str,
                             frame_length: int = 10):
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
class LoadNpyFile(BaseTransform):

    def __init__(self, frame_length=10, label_idxs=[0, 9]) -> None:
        self.frame_length = frame_length
        self.label_idxs = label_idxs

    def transform(self, results: Dict) -> Optional[Dict]:
        img_path = results['img_path']
        anno_path = results['seg_map_path']
        imgs, masks, ef = load_video_and_mask_file(img_path, anno_path,
                                                   self.frame_length)

        results['img'] = imgs
        results['ori_imgs'] = imgs
        results['masks'] = masks
        results['frame_length'] = self.frame_length
        results['label_idxs'] = self.label_idxs
        results['ef'] = ef
        results['imgs_shape'] = imgs.shape
        results['img_shape'] = imgs.shape[2:]
        results['ori_shape'] = imgs.shape[2:]

        return results


@TRANSFORMS.register_module()
class PackSegMultiInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label',
                            'frame_length', 'label_idxs')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if 'img' in results:
            imgs = [to_tensor(img).contiguous() for img in results['img']]
            packed_results['inputs'] = imgs

        data_samples = []
        if 'masks' in results:
            for mask in results['masks']:
                data_sample = SegEchoDataSample()
                mask = to_tensor(mask.astype(np.int64))
                mask.unsqueeze_(0)
                gt_sem_seg_data = dict(data=mask)
                data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
                data_samples.append(data_sample)

        if 'masks_edge' in results:
            for i in range(len(results['masks_edge'])):
                mask_edge = results['masks_edge'][i]
                mask_edge = to_tensor(mask_edge.astype(np.int64))
                mask_edge.unsqueeze_(0)
                gt_edge_data = dict(data=mask_edge)
                data_samples[i].set_data(
                    dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        if 'ef' in results:
            img_meta['ef'] = to_tensor(results['ef'].astype(np.float32))

        if 'label_idxs' in img_meta:
            for i in range(len(img_meta['label_idxs'])):
                data_sample = data_samples[i]
                img_meta['frame_idx'] = img_meta['label_idxs'][i]
                data_sample.set_metainfo(img_meta)
                ori_img_data = dict(data=results['img'][img_meta['frame_idx'],
                                                        ...])
                data_sample.set_data(dict(ori_img=PixelData(**ori_img_data)))

        packed_results['data_samples'] = data_samples
        return packed_results


@TRANSFORMS.register_module()
class TestPackSegMultiInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label',
                            'frame_length', 'label_idxs')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        data_samples = []
        if 'img' in results:
            imgs = [to_tensor(img) for img in results['img']]
            packed_results['inputs'] = imgs
            for img in results['img']:
                data_sample = SegEchoDataSample()
                ori_img_data = dict(data=img)
                data_sample.set_data(dict(ori_img=PixelData(**ori_img_data)))
                data_samples.append(data_sample)

        assert 'label_idxs' in results
        if 'masks' in results:
            for i in range(len(results['masks'])):
                mask = results['masks'][i]
                mask = to_tensor(mask.astype(np.int64))
                mask.unsqueeze_(0)
                gt_sem_seg_data = dict(data=mask)
                data_samples[results['label_idxs'][i]].set_data(
                    dict(gt_sem_seg=PixelData(**gt_sem_seg_data)))

        if 'masks_edge' in results:
            for i in range(len(results['masks_edge'])):
                mask_edge = results['masks_edge'][i]
                mask_edge = to_tensor(mask_edge.astype(np.int64))
                mask_edge.unsqueeze_(0)
                gt_edge_data = dict(data=mask_edge)
                data_samples[results['label_idxs'][i]].set_data(
                    dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        if 'ef' in results:
            img_meta['ef'] = to_tensor(results['ef'].astype(np.float32))

        if 'label_idxs' in img_meta:
            for i, data_sample in enumerate(data_samples):
                img_meta['frame_idx'] = i
                data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_samples
        return packed_results


@TRANSFORMS.register_module()
class LoadVideoAndAnnoFromFile(BaseTransform):
    '''Load n frames video from .npy file
        args:
            frame_length: (int) least 2
    '''

    def __init__(self, frame_length=10) -> None:
        self.frame_length = frame_length

    def transform(self, results: dict) -> Optional[dict]:
        img_path = results['img_path']
        anno_path = results['seg_map_path']
        imgs, masks, ef = load_video_and_mask_file(img_path, anno_path,
                                                   self.frame_length)

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
        for mask in results['masks']:
            results['gt_seg_map'] = mask
            super().transform(results)
            mask_edge.append(results['gt_edge_map'])
        results['masks_edge'] = mask_edge
        return results


@TRANSFORMS.register_module()
class VideoRandomResize(RandomResize):

    def transform(self, results: dict) -> dict:
        return super().transform(results)


@TRANSFORMS.register_module()
class VideoResize(Resize):

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            results['img'] = results['img'].transpose(0, 2, 3, 1)
            imgs = []
            for i in range(results['img'].shape[0]):
                # if self.keep_ratio:
                #     img, scale_factor = mmcv.imrescale(
                #         results['img'],
                #         results['scale'],
                #         interpolation=self.interpolation,
                #         return_scale=True,
                #         backend=self.backend)
                #     # the w_scale and h_scale has minor difference
                #     # a real fix should be done in the mmcv.imrescale in the future
                #     new_h, new_w = img.shape[:2]
                #     h, w = results['img'].shape[:2]
                #     w_scale = new_w / w
                #     h_scale = new_h / h
                # else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'][i],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                imgs.append(img)
            results['img'] = np.array(imgs).transpose(0, 3, 1, 2)
            results['img_shape'] = imgs[0].shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('masks', None) is not None:
            masks = []
            for i in range(results['masks'].shape[0]):
                gt_seg = mmcv.imresize(
                    results['masks'][i],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
                masks.append(gt_seg)
            results['masks'] = np.array(masks)

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results


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
            if 'img' in results:
                imgs = results['img'].transpose(0, 2, 3, 1)
                for i, img in enumerate(imgs):
                    imgs[i] = mmcv.imflip(img=img, direction=cur_dir)
                results['img'] = imgs.transpose(0, 3, 1, 2)
            # filp masks
            if 'masks' in results:
                masks = results['masks'].transpose(1, 2, 0)
                masks = mmcv.imflip(img=masks, direction=cur_dir)
                results['masks'] = masks.transpose(2, 0, 1)
            # filp masks_edge
            if 'masks_edge' in results:
                for i, mask_edge in enumerate(results['masks_edge']):
                    results['masks_edge'][i] = mmcv.imflip(
                        img=mask_edge, direction=cur_dir)
        return results


@TRANSFORMS.register_module()
class VideoRandomCrop(RandomCrop):

    def transform(self, results: dict) -> dict:
        return super().transform(results)


@TRANSFORMS.register_module()
class VideoPhotoMetricDistortion(PhotoMetricDistortion):

    def hue(self, imgs: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            imgs (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if random.randint(2):
            for i, img in enumerate(imgs):
                img = mmcv.bgr2hsv(img)
                img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(
                    -self.hue_delta, self.hue_delta)) % 180
                imgs[i] = mmcv.hsv2bgr(img)
        return imgs

    def saturation(self, imgs: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            imgs (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if random.randint(2):
            for i, img in enumerate(imgs):
                img = mmcv.bgr2hsv(img)
                img[:, :, 1] = self.convert(
                    img[:, :, 1],
                    alpha=random.uniform(self.saturation_lower,
                                         self.saturation_upper))
                imgs[i] = mmcv.hsv2bgr(img)
        return imgs

    def transform(self, results: dict) -> dict:
        imgs = results['img']
        imgs = imgs.transpose(0, 2, 3, 1)
        # random brightness
        imgs = self.brightness(imgs)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            imgs = self.contrast(imgs)

        # random saturation
        imgs = self.saturation(imgs)

        # random hue
        imgs = self.hue(imgs)

        # random contrast
        if mode == 0:
            imgs = self.contrast(imgs)

        results['img'] = imgs.transpose(0, 3, 1, 2)
        return results
