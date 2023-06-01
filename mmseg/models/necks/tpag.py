import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.backbones.pidnetv2 import Bag

@MODELS.register_module()
class TPag(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        pass