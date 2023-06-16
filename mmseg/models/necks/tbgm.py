
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import build_activation_layer, build_norm_layer


from mmseg.registry import MODELS


@MODELS.register_module()
class TemporalBoundaryGuidedMoudle(nn.Module):
    def __init__(self,                  
            in_channels: int,
            channels: int,
            kernel_size: int,
            norm_cfg: dict(type='BN'),
            act_cfg: dict(type='ReLU', inplace=True)):
        super().__init__()
        # assert in_channels == channels * 2
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            order=('norm', 'act', 'conv')
        )
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)
    
    def forward(self, inputs):
        p, sem, boundary = inputs
        _, _, h, w = boundary.shape
        boundary = boundary.view(self.frame_length, self.batchsize, -1, h, w)
        b_concat = torch.cat((boundary[:-1,...], boundary[1:,...]), dim=2)
        b_concat = b_concat.view((self.frame_length-1)*self.batchsize, -1, h, w)
        b_concat = self.conv(b_concat)

        sem = sem.view(self.frame_length, self.batchsize, -1, h, w)
        s_1 = sem[:-1,...].view((self.frame_length-1)*self.batchsize, -1, h, w)
        s_2 = sem[1:,...].view((self.frame_length-1)*self.batchsize, -1, h, w)

        b_concat = b_concat * s_1 + s_2
        b_concat = b_concat.view(self.frame_length-1, self.batchsize, -1, h, w)
        sem[:-1,...] = b_concat

        sem = sem.view(self.frame_length*self.batchsize, -1, h, w)
        boundary = boundary.view(self.frame_length*self.batchsize, -1, h, w)
        sem = self.norm(sem)
        sem = self.act(sem)
        return (p, sem, boundary)

