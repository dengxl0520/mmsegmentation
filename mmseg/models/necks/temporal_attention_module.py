import numpy as np
from einops import rearrange, repeat
from typing import Callable, Dict, List, Optional, Tuple, Union

import copy
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from mmcv.cnn import ConvModule
from mmengine.model.weight_init import caffe2_xavier_init
from mmseg.registry import MODELS

from .temporal_neck.pos_embeddings import PositionEmbeddingSine, PositionEmbeddingSine3D
from .temporal_neck.temporal_attention import TemporalAttentionLayer, get_patch_mask_indices
from .temporal_neck.temporal_neck_ops.modules import MSDeformAttn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 temporal_attn_ksize_offset=0):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        # self.patches_per_dim = patches_per_dim
        self.temporal_attn_ksize_offset = temporal_attn_ksize_offset

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout, activation=activation,
            n_levels=num_feature_levels, n_heads=nhead, n_points=enc_n_points
        )

        temporal_layer = TemporalAttentionLayer(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout, activation=activation, n_heads=nhead
        )

        self.encoder = MSDeformAttnTransformerEncoder(
            encoder_layer=encoder_layer, temporal_layer=temporal_layer, num_layers=num_encoder_layers
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.level_embed_3d = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)
        normal_(self.level_embed_3d)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, pos_embeds_3d):
        # batch_sz, clip_len = srcs[0].shape[:2]
        patch_mask_indices = get_patch_mask_indices(srcs, ksize_offset=self.temporal_attn_ksize_offset)  # List[num_lvls, [num_patches, patch_size]]
        srcs = [rearrange(x, "B T C H W -> (B T) C H W") for x in srcs]
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        lvl_pos_embed_3d_flatten = []

        for lvl, (src, mask, pos_embed, pos_embed_3d) in \
                enumerate(zip(srcs, masks, pos_embeds, pos_embeds_3d)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # [B*T, H*W, C]
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

            pos_embed_3d = rearrange(pos_embed_3d, "B T C H W -> B T (H W) C")
            lvl_pos_embed_3d = pos_embed_3d + self.level_embed_3d[lvl].view(1, 1, 1, -1)
            lvl_pos_embed_3d_flatten.append(lvl_pos_embed_3d)

        src_flatten = torch.cat(src_flatten, 1)  # [B*T, L*H*W*, C]
        mask_flatten = torch.cat(mask_flatten, 1)  # [B*T, L*H*W]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        lvl_pos_embed_3d_flatten = torch.cat(lvl_pos_embed_3d_flatten, 2)
        patch_mask_indices = torch.cat(patch_mask_indices, 1)  # [num_patches, patch_size_over_all_levels]

        # encoder
        memory = self.encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=lvl_pos_embed_flatten,
            padding_mask=mask_flatten,
            pos_3d=lvl_pos_embed_3d_flatten,
            patch_mask_indices=patch_mask_indices,
        )

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @autocast(enabled=False)
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        # print("Deformatnn:", src.dtype, pos.dtype)
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, temporal_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.temporal_layers = _get_clones(temporal_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask,
                pos_3d, patch_mask_indices):

        output = src
        # reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        for _, layer_temporal in enumerate(self.temporal_layers):
            output = layer_temporal(src=output, pos=pos_3d, patch_mask_indices=patch_mask_indices)

        # for _, (layer, layer_temporal) in enumerate(zip(self.layers, self.temporal_layers)):
        #     output_1 = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        #     output = layer_temporal(src=output_1, pos=pos_3d, patch_mask_indices=patch_mask_indices)
            # print(f"Sanity test: {torch.all(output == output_1)}")

        return output

@MODELS.register_module()
# @GlobalRegistry.register("PixelDecoder", "m2f_timesformer")
class TemporalNeckSimple(nn.Module):
    # @configurable
    def __init__(
            self,
            input_shape: Dict,
            transformer_dropout: float = 0.0,
            transformer_nheads: int = 8,
            transformer_dim_feedforward: int = 1024,
            transformer_enc_layers: int = 6,
            conv_dim: int = 256,
            mask_dim: int = 256,
            norm: Optional[Union[str, Callable]] = 'GN',
            # deformable transformer encoder args
            transformer_in_features: List[str] = ['res3', 'res4', 'res5'],
            common_stride: int = 4,
            temporal_attn_patches_per_dim: int = 8,
            temporal_attn_ksize_offset: int = 1
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        self.input_shape = input_shape

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]

        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            temporal_attn_ksize_offset=temporal_attn_ksize_offset
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.pe_layer_3d = PositionEmbeddingSine3D(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = ConvModule(
            conv_dim,
            mask_dim,
            kernel_size=1,
            act_cfg=None
        )

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        # 改写成mmseg的conv2d和norm
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = ConvModule(
                in_channels,
                conv_dim,
                kernel_size=1,
                bias=use_bias,
                norm_cfg=dict(type=norm, num_groups=conv_dim),
                act_cfg=None
            )
            output_conv = ConvModule(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm_cfg=dict(type=norm, num_groups=conv_dim),
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.is_3d = True
        self.fmap_multiple_of = 64

    def init_weights(self) -> None:
        caffe2_xavier_init(self.mask_features)
        caffe2_xavier_init(self.output_convs)
        caffe2_xavier_init(self.lateral_convs)

    # @autocast(enabled=False)
    def forward(self, features):
        temp = {}
        for i, k in enumerate(self.input_shape):
            x = features[i]
            temp[k]  = rearrange(x, "(B T) C H W -> B T C H W", B=self.batchsize, T=self.frame_length)
        features = temp

        srcs = []
        pos = []
        pos_3d = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            # x = features[f].float()  # deformable detr does not support half precision
            x = features[f]
            pos_3d.append(self.pe_layer_3d(x, fmt='btchw'))

            # input_proj is Conv2d
            batch_sz, clip_len = x.shape[:2]
            x = rearrange(x, "B T C H W -> (B T) C H W")
            pos.append(self.pe_layer(x))

            x = self.input_proj[idx](x)
            srcs.append(rearrange(x, "(B T) C H W -> B T C H W", B=batch_sz, T=clip_len))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos, pos_embeds_3d=pos_3d)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # merge batch and temporal dimensions for remaining forward pass
        features = {
            key: rearrange(f, "B T C H W -> (B T) C H W") for key, f in features.items()
        }

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            # x = features[f].float()
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        # return self.mask_features(out[-1]), out[0], multi_scale_features
        # print(out[-1].min(), out[-1].max())
        multi_scale_features.append(self.mask_features(out[-1]))
        
        # print(multi_scale_features[-1].min(), multi_scale_features[-1].max())
        return multi_scale_features[::-1]


