import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

@MODELS.register_module()
class ConsistencyLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0) -> None:
        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target = F.softmax(target, dim=2)
        target1, target2 = target[:-1], target[1:]
        f, bs, c, h, w = target1.shape
        target1, target2 = target1.contiguous().view(-1,c,h,w), target2.contiguous().view(-1,c,h,w)
        mse = F.mse_loss(
            input=target2, target=target1, reduction=reduction)
        loss1 = mse.sum() / (f*bs*c)
        epsilon = 1e-8
        target1_smooth = target1 + epsilon
        target2_smooth = target2 + epsilon
        kl =  F.kl_div(
            input=target2_smooth.log(),
            target=target1_smooth.log(),
            reduction=reduction,
            log_target=True)
        loss2 = kl.mean()
        return self.loss_weight * self.t * (loss1 + loss2)

@MODELS.register_module()
class KLConsistencyLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0) -> None:
        super(KLConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target = F.softmax(target, dim=2)
        target1 = target[:-1]
        target2 = target[1:]
        kl =  F.kl_div(
            input=target2,
            target=target1,
            reduction=reduction,
            log_target=True)
        loss = kl.sum()
        return self.loss_weight * self.t * loss

@MODELS.register_module()
class MSEConsistencyLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0) -> None:
        super(MSEConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target = F.softmax(target, dim=2)
        target_detach = target.detach()
        mse1 = F.mse_loss(input=target[1:-1], target=target_detach[2:], reduction=reduction)
        mse2 = F.mse_loss(input=target[1:-1], target=target_detach[:-2], reduction=reduction)
        loss = (mse1+mse2).mean()
        if hasattr(self, 't'):
            return self.loss_weight * self.t * loss
        else:
            return self.loss_weight * loss
        
@MODELS.register_module()
class MSEConsistencyLoss1(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0) -> None:
        super(MSEConsistencyLoss1, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target = F.softmax(target, dim=2)
        target_detach = target.detach()
        mse1 = F.mse_loss(input=target[1:-1], target=target_detach[2:], reduction=reduction)
        mse2 = F.mse_loss(input=target[1:-1], target=target_detach[:-2], reduction=reduction)
        loss = abs(mse1.mean()-mse2.mean())
        if hasattr(self, 't'):
            return self.loss_weight * self.t * loss
        else:
            return self.loss_weight * loss
        
@MODELS.register_module()
class AbsMSEConsistencyLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0) -> None:
        super(AbsMSEConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                target,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target = F.softmax(target, dim=2)
        target_detach = target.detach()
        mse1 = F.mse_loss(input=target[1:-1], target=target_detach[2:], reduction=reduction)
        mse2 = F.mse_loss(input=target[1:-1], target=target_detach[:-2], reduction=reduction)
        loss = abs(mse1.mean()-mse2.mean())
        if hasattr(self, 't'):
            return self.loss_weight * self.t * loss
        else:
            return self.loss_weight * loss