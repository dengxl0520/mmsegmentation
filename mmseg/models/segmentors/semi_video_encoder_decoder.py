import torch 
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import List, Optional

from torch import Tensor

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

    def forward(self, inputs: Tensor, data_samples, mode: str = 'tensor'):
        assert data_samples is not None
        # init semi-info
        self.frame_length = data_samples[0].frame_length
        self.label_idxs = data_samples[0].label_idxs
        self.batchsize = len(data_samples) // len(self.label_idxs)
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

        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        # visualizer = Visualizer.get_current_instance()
        # for level, feats in enumerate(x):
        #     for idx, feat in enumerate(x[level]):
        #         ori_img = data_samples[idx].get('ori_img').data.transpose(1,2,0)
        #         size = data_samples[idx].get('img_shape')
        #         # feat = 1 - feat 
        #         drawn_img = visualizer.draw_featmap(feat.sigmoid(), ori_img, channel_reduction='select_max', resize_shape=size)
        #         visualizer.add_image('feat'+ str(level) + str(idx) + 'frame' , drawn_img)
        # draw seg_logits
        # for idx, seg_logits_frame in enumerate(seg_logits):
        #     ori_img = data_samples[idx].get('ori_img').data.transpose(1,2,0)
        #     size = data_samples[idx].get('img_shape')
        #     seg_logit_img = visualizer.draw_featmap(seg_logits_frame, ori_img, channel_reduction='select_max', resize_shape=size)
        #     visualizer.add_image('seg_logits_' + str(idx) + 'frame', seg_logit_img) 
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
        x = [i[self.sup_feature_idxs,...] for i in x]

        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return self.postprocess_result(seg_logits, data_samples)
    
    def _semi_forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x = self.semi_extract_feat(inputs)
        x = [i[self.sup_feature_idxs,...] for i in x]

        return self.decode_head.forward(x)