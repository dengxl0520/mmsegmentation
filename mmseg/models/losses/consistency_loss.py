import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

def kl():
    pass

@MODELS.register_module()
class MSEConsistencyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0) -> None:
        super(MSEConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target1 = target[:-1]
        target2 = target[1:]
        loss = self.loss_weight * F.mse_loss(
            input=target1, target=target2, reduction=reduction)
        return loss
