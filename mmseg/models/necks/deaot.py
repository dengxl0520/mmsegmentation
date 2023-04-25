# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from mmseg.registry import MODELS


@MODELS.register_module()
class DeAOT(nn.Module):

