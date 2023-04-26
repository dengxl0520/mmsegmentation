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
                 input_type: str = ''):
        super().__init__(backbone, decode_head, neck, auxiliary_head,
                         train_cfg, test_cfg, data_preprocessor, pretrained,
                         init_cfg)
        self.input_type = input_type

    def forward(self, inputs: Tensor, data_samples, mode: str = 'tensor'):
        '''The unified entry for a forward process in both training and test.
        
        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        '''
        if self.input_type == 'img':
            super().forward(inputs, data_samples, mode)
        elif self.input_type == 'video':
            # frames
            frame_length = inputs.shape[1]
            results = dict() if mode == 'loss' else []
            for i in range(frame_length):
                if i == 0 or i == frame_length - 1:
                    inputframe = inputs[:, i, :, :, :]
                    inputframe.squeeze_(dim=1)
                    data_samples_new = [
                        data_sample.clone() for data_sample in data_samples
                    ]
                    if i == 0:
                        for data_sample in data_samples_new:
                            data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.frame1
                            if 'gt_edge_map' in data_sample:
                                data_sample.gt_edge_map.data = data_sample.gt_edge_map.frame1_edge
                            data_sample.set_metainfo(dict(frameidx='1'))
                    else:
                        for data_sample in data_samples_new:
                            data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.frame2
                            if 'gt_edge_map' in data_sample:
                                data_sample.gt_edge_map.data = data_sample.gt_edge_map.frame2_edge
                            data_sample.set_metainfo(dict(frameidx='2'))

                    if mode == 'loss':
                        result = self.loss(inputframe, data_samples_new)
                        results.update(result)
                    elif mode == 'predict':
                        result = self.predict(inputframe, data_samples_new)
                        results.append(*result)
                    elif mode == 'tensor':
                        result = self._forward(inputframe, data_samples_new)
                        results.append(*result)
                    else:
                        raise RuntimeError(
                            f'Invalid mode "{mode}". '
                            'Only supports loss, predict and tensor mode')
            return results
