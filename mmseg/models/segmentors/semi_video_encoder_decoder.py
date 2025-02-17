import torch 
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import List, Optional

from torch import Tensor
import time

from mmengine.visualization import Visualizer
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils import resize


@MODELS.register_module()
class SemiVideoEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(backbone, decode_head, neck, auxiliary_head,
                         train_cfg, test_cfg, data_preprocessor, pretrained,
                         init_cfg)

    def set_train_cfg(self, data_samples, batchsize = 0):
        # init semi-info
        self.frame_length = data_samples[0].frame_length
        self.label_idxs = data_samples[0].label_idxs
        if batchsize > 0:
            self.batchsize = batchsize
        else:
            self.batchsize = len(data_samples) // len(self.label_idxs)
        # if self.training:
        #     self.batchsize = len(data_samples) // len(self.label_idxs)
        # else:
        #     self.batchsize = len(data_samples) // self.frame_length
        self.sup_feature_idxs = []
        for i in range(self.batchsize):
            self.sup_feature_idxs.extend(
                [label_idx + self.frame_length * i for label_idx in self.label_idxs])

        if self.with_neck:
            self.neck.sup_feature_idxs = self.sup_feature_idxs
            self.neck.frame_length = self.frame_length
            self.neck.batchsize = self.batchsize
            
        if self.with_decode_head:
            self.decode_head.sup_feature_idxs = self.sup_feature_idxs
            self.decode_head.frame_length = self.frame_length
            self.decode_head.batchsize = self.batchsize

        if self.with_auxiliary_head:
            self.auxiliary_head.sup_feature_idxs = self.sup_feature_idxs
            self.auxiliary_head.frame_length = self.frame_length
            self.auxiliary_head.batchsize = self.batchsize

    def forward(self, inputs: Tensor, data_samples, mode: str = 'tensor'):
        assert data_samples is not None
        
        self.set_train_cfg(data_samples)

        if mode == 'loss':
            return self.semi_loss(inputs, data_samples)
        elif mode == 'predict':
            return self.semi_predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._semi_forward(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". '
                'Only supports loss, predict and tensor mode')

    def draw_featmap(self, x, data_samples, mode, alpha, channel_reduction='squeeze_mean'):
        '''
            channel_reduction: 'squeeze_mean' or 'select_max'
        '''
        visualizer = Visualizer.get_current_instance()
        if mode == 'feat':
            for level, feats in enumerate(x):
                for idx, feat in enumerate(x[level]):
                    ori_img = data_samples[idx].get('ori_img').data.transpose(1,2,0)
                    size = data_samples[idx].get('img_shape')
                    # feat = 1 - feat 
                    drawn_img = visualizer.draw_featmap(
                        feat,
                        ori_img,
                        channel_reduction=channel_reduction,
                        resize_shape=size,
                        alpha=alpha) # alpha
                    visualizer.add_image('feat'+ str(level) + str(idx) + 'frame' , drawn_img)
        elif mode == 'logit':
            seg_logits = x
            # draw seg_logit
            for idx, seg_logits_frame in enumerate(seg_logits):
                ori_img = data_samples[idx].get('ori_img').data.transpose(1,2,0)
                size = data_samples[idx].get('img_shape')
                filename = data_samples[idx].get('img_path')
                filename = filename.split('/')[-1].split('.')[0]
                seg_logit_img = visualizer.draw_featmap(
                    seg_logits_frame,
                    ori_img, 
                    channel_reduction=channel_reduction,
                    resize_shape=size, 
                    alpha=alpha)
                visualizer.add_image(f'seg_logits_{filename}_{idx}_frame', seg_logit_img) 

    def draw_ori_img(self, data_samples):
        import cv2
        import numpy as np

        filename = data_samples[0].get('img_path')
        filename = filename.split('/')[-1].split('.')[0]

        for idx, data_sample in enumerate(data_samples):
            ori_img = data_sample.get('ori_img').data
            ori_img = np.transpose(ori_img, (1,2,0)) # [3,112,112] => [112,112,3]
            cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename=f'{filename}_ori_frame_{idx}.png', img=ori_img)

    def draw_pred(self, data_samples):
        import cv2
        import numpy as np

        filename = data_samples[0].get('img_path')
        filename = filename.split('/')[-1].split('.')[0]

        for idx, data_sample in enumerate(data_samples):
            pred_sem_seg = data_sample.get('pred_sem_seg').data
            pred_sem_seg = pred_sem_seg.cpu().numpy().astype(np.uint8)
            pred_sem_seg[pred_sem_seg == 1] = 255
            pred_sem_seg = np.transpose(pred_sem_seg, (1,2,0))
            # cv2.cvtColor(pred_sem_seg, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename=f'{filename}_predmask_frame_{idx}.png', img=pred_sem_seg)



    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        x = self.semi_extract_feat(inputs)


        '''
        pred = self.decode_head(x,data_samples)[-1]
        pred[pred < 0] = 0

        filename = data_samples[0].get('img_path')
        filename = filename.split('/')[-1].split('.')[0]
        if filename == '0X1D24CEA29A24560B':
            self.draw_featmap(pred, data_samples, mode='logit', alpha=0.4)
            # self.draw_ori_img(data_samples)

            seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
            results = self.postprocess_result(seg_logits, data_samples)

            # self.draw_pred(data_samples=results)
        '''

        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return self.postprocess_result(seg_logits, data_samples)
    
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x = self.semi_extract_feat(inputs)

        return self.decode_head.forward(x, data_samples)
     
    def semi_extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def semi_loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.semi_extract_feat(inputs)

        losses = dict()
        if self.with_auxiliary_head:
            loss_aux = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
            
        x = [i[self.sup_feature_idxs,...] for i in x]

        loss_decode = self.decode_head.loss(x, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def semi_predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """

        # start_time = time.time()

        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        x = self.semi_extract_feat(inputs)

        '''save feature
        import os
        import numpy as np
        save_path = 'work_dirs/feature1/'
        filename = data_samples[0].get('img_path')
        filename = filename.split('/')[-1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        feature = x[0].clone().cpu().numpy()
        np.save(save_path + filename,  feature)
        '''
        
        x = [i[self.sup_feature_idxs,...] for i in x]

        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        
        # results = self.postprocess_result(seg_logits, data_samples)
        
        # end_time = time.time()

        # elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time} seconds")
        # return results

        return self.postprocess_result(seg_logits, data_samples)
    
    def _semi_forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x = self.semi_extract_feat(inputs)
        x = [i[self.sup_feature_idxs,...] for i in x]

        return self.decode_head.forward(x, data_samples)