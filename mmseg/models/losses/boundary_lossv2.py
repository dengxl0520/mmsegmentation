import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Iterable, List, Tuple, Set, cast
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance

from mmseg.registry import MODELS

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def one_hot2dist(seg: Tensor, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    seg = seg.detach().cpu().numpy()
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res
    
@MODELS.register_module()
class BoundaryLossV2(nn.Module):
    """Boundary loss from `Boundary loss for highly unbalanced segmentation`.

    This function is modified from
    `https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py#L96`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_boundary',
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        # self.idc: List[int] = kwargs["idc"]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")


    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Forward function.
        Args:
            pred (Tensor): Predictions of the head.
            dist_map (Tensor): dist_map of Ground truth.

        Returns:
            Tensor: Loss tensor.
        """
        # assert simplex(probs)
        # assert not one_hot(dist_maps)
        pred = pred.squeeze()
        pc = pred.type(torch.float32)
        dist_map = one_hot2dist(gt)
        dist_map = torch.tensor(dist_map, device=pc.device)
        dc = dist_map.type(torch.float32)

        multipled = einsum("bhw,bhw->bhw", pc, dc)

        loss = multipled.mean()

        return self.loss_weight * loss


    @property
    def loss_name(self):
        return self.loss_name_
