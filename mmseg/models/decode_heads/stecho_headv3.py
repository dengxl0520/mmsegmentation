import copy
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import Conv2d
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from mmdet.registry import TASK_UTILS
from mmdet.models.utils.point_sample import point_sample, get_uncertain_point_coords_with_randomness
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean, InstanceList, ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from .mask2former_head import Mask2FormerHead
from ..builder import build_loss

import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import ModuleList
from mmengine.visualization import Visualizer
from mmdet.models.layers.transformer.detr_layers import DetrTransformerDecoder, DetrTransformerDecoderLayer

class BoundaryTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            BoundaryAttentionLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]

class BoundaryAttentionLayer(DetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.boundary_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)


    def forward(self, 
                query: Tensor, 
                key: Tensor = None, 
                value: Tensor = None, 
                query_pos: Tensor = None, 
                key_pos: Tensor = None, 
                self_attn_mask: Tensor = None, 
                corss_attn_mask: Tensor = None,
                boundary_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None, 
                **kwargs) -> Tensor:
        
        tgt_boundary = self.boundary_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=boundary_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        # tgt_cross_attn = self.cross_attn(
        #     query=query,
        #     key=key,
        #     value=value,
        #     query_pos=query_pos,
        #     key_pos=key_pos,
        #     attn_mask=corss_attn_mask,
        #     key_padding_mask=key_padding_mask,
        #     **kwargs)
        # query = self.norms[0](tgt_boundary + tgt_cross_attn)
        query = self.norms[0](tgt_boundary)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query



@MODELS.register_module()
class STEchoHeadv3(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_decode: Dict = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 align_corners = False,
                 ignore_index = 255,
                 threshold=0.5,
                 **kwargs) -> None:
        super().__init__()
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = BoundaryTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.align_corners = align_corners
        self.out_channels = out_channels
        self.ignore_index = ignore_index
        self.threshold = threshold

        # build loss
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples).squeeze()

        all_mask_preds = self(x, batch_data_samples)

        if 'pad_shape' in batch_data_samples[0]:
            size = batch_data_samples[0].get('pad_shape')
        else:
            size = batch_data_samples[0].get('img_shape')

        # losses
        losses_decode = self.loss_decode
        losses_ce, losses_dice = [], []
        for mask in all_mask_preds:
            seg_logits = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
            seg_logits = seg_logits.sigmoid()
            loss_ce = losses_decode[0](
                    seg_logits,
                    seg_label,
                    ignore_index=self.ignore_index)
            loss_dice = losses_decode[1](
                    seg_logits,
                    seg_label,
                    ignore_index=self.ignore_index)
            
            losses_ce.append(loss_ce)
            losses_dice.append(loss_dice)

        loss_dict = dict()
        loss_dict['loss_ce'] = losses_ce[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        num_dec_layer = 0
        for loss_mask_i, loss_dice_i in zip(losses_ce[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_ce'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict
    
    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_mask_preds = self(x, batch_data_samples)

        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        seg_logits = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        
        seg_logits = seg_logits.sigmoid()
        return seg_logits
      
    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        # cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        cross_attn_mask = attn_mask.sigmoid() < 0.5
        in_attn_mask = attn_mask.sigmoid() < 0.4 
        out_attn_mask = attn_mask.sigmoid() > 0.6 
        boundary_attn_mask = in_attn_mask | out_attn_mask
        cross_attn_mask = cross_attn_mask.detach()
        boundary_attn_mask = boundary_attn_mask.detach()

        return mask_pred, cross_attn_mask, boundary_attn_mask

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        # batch_size = len(batch_img_metas)
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)

        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        mask_pred_list = []
        mask_pred, cross_attn_mask, boundary_attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            cross_attn_mask[torch.where(
                cross_attn_mask.sum(-1) == cross_attn_mask.shape[-1])] = False
            boundary_attn_mask[torch.where(
                boundary_attn_mask.sum(-1) == boundary_attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                corss_attn_mask=cross_attn_mask,
                boundary_attn_mask=boundary_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            mask_pred, cross_attn_mask, boundary_attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            mask_pred_list.append(mask_pred)

        return mask_pred_list


