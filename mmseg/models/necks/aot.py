import torch.nn as nn
from mmcv.cnn import build_norm_layer

from mmseg.registry import MODELS


@MODELS.register_module()
class AOT(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        pass
