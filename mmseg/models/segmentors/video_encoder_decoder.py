import torch 
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import List, Optional

from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)


@MODELS.register_module()
class VideoEncoderDecoder(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 input_type: str = 'video',
                 supervised: str = 'sup'):
        super().__init__(backbone, decode_head, neck, auxiliary_head,
                         train_cfg, test_cfg, data_preprocessor, pretrained,
                         init_cfg)
        assert input_type == 'img' or input_type == 'video'
        assert supervised == 'sup' or supervised == 'semisup'
        self.input_type = input_type
        self.supervised = supervised


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

        if self.input_type == 'img' and self.supervised == 'sup':
            return super().forward(inputs, data_samples, mode)
        elif self.input_type == 'video' and self.supervised == 'sup':
            return super().forward(inputs, data_samples, mode)
        elif self.input_type == 'video' and self.supervised == 'semisup':
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

    def cut_before_backbone(self, inputs: Tensor) -> List[Tensor]:
        # cut inputs: only use frames with anno
        self.frame_length = 2
        if len(self.sup_feature_idxs) != len(inputs):
            inputs = inputs[self.sup_feature_idxs, ...]
        return inputs
    
    def cut_after_backbone(self, x: Tensor) -> List[Tensor]:
        # cut inputs: only use features with anno
        if isinstance(x,list) or isinstance(x,tuple):
            x = [f[self.sup_feature_idxs, ...]for f in x]
        else:
            x = x[self.sup_feature_idxs, ...]
        return x
    
    def semi_neck_forward(self, x: Tensor):
        if self.with_neck:
            assert len(x) == 3 and isinstance(x,tuple)
            temp_p, out, temp_d = x
            if len(out) == 3 and isinstance(out, tuple):
                # no dfm
                # multi_gpm forward
                x_p, x_i, x_d = out
                temp = []
                for i in range(self.frame_length):
                    frame_idx = [i + j*self.frame_length for j in range(self.batchsize)]
                    _p = x_p[frame_idx,...]
                    _i = x_i[frame_idx,...]
                    _d = x_d[frame_idx,...]
                    after_neck_x = self.neck([_p,_i,_d])
                    temp.append(after_neck_x)
            elif isinstance(out, Tensor):
                # after dfm
                # single_gpm forward
                temp = []
                for i in range(self.frame_length):
                    frame_idx = [i + j*self.frame_length for j in range(self.batchsize)]
                    _out = out[frame_idx,...]
                    after_neck_x = self.neck([_out])
                    temp.append(after_neck_x[0])
            # clear_memories
            self.neck.clear_memories()
            # collect outputs
            outputs = []
            for i in range(self.batchsize):
                for j in range(self.frame_length):
                    outputs.append(temp[j][i,...])
            outputs = torch.stack(outputs)
            x = (temp_p, outputs, temp_d) if self.training else outputs

        # only use frames with anno
        if self.frame_length == len(self.label_idxs) :
            sup_featrue = x
        elif isinstance(x, tuple) or isinstance(x, list):
            if len(self.sup_feature_idxs) != len(x[0]): 
                sup_featrue = [f[self.sup_feature_idxs, ...] for f in x]
        elif len(self.sup_feature_idxs) != len(x):
            sup_featrue = x[self.sup_feature_idxs, ...]            
        return sup_featrue 
            
    def semi_extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # inputs = self.cut_before_backbone(inputs)
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.semi_neck_forward(x)
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

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

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
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return self.postprocess_result(seg_logits, data_samples)

    def _semi_forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.semi_extract_feat(inputs)
        return self.decode_head.forward(x)
