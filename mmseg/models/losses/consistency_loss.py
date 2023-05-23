import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

@MODELS.register_module()
class ConsistencyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0) -> None:
        super(ConsistencyLoss, self).__init__()
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
        loss1 = self.loss_weight * F.mse_loss(
            input=target2, target=target1, reduction=reduction)
        loss2 = self.loss_weight * F.kl_div(
            input=F.log_softmax(target2),
            target=F.log_softmax(target1),
            reduction=reduction,
            log_target=True)
        return loss1 + loss2

@MODELS.register_module()
class KLConsistencyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0) -> None:
        super(KLConsistencyLoss, self).__init__()
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
        loss = self.loss_weight * F.kl_div(
            input=F.log_softmax(target2),
            target=F.log_softmax(target1),
            reduction=reduction,
            log_target=True)
        return loss

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
            input=target2, target=target1, reduction=reduction)
        return loss
