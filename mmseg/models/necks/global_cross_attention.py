
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import ModuleList
from mmdet.models.layers.transformer.detr_layers import DetrTransformerDecoderLayer, DeformableDetrTransformerEncoder


class GlobalCrossTransformerDecoder(DeformableDetrTransformerEncoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GlobalCrossAttentionLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]


class GlobalCrossAttentionLayer(DetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_foreground = MultiheadAttention(**self.cross_attn_cfg)
        self.cross_attn_background = MultiheadAttention(**self.cross_attn_cfg)
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
                cross_attn_mask: Tensor = None, 
                key_padding_mask: Tensor = None, 
                **kwargs) -> Tensor:
        
        tgt_foreground = self.cross_attn_foreground(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        tgt_background = self.cross_attn_background(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[0](tgt_foreground + tgt_background)
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
