import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Type
import numpy as np 


class CrossFrameAttention(nn.Module):
    
    def __init__(self, input_dim, num_heads, batch_size, max_fr):
        super(CrossFrameAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=True)
        self.video_length = max_fr
        head_dim = input_dim // self.num_heads
        self.scale = head_dim**-0.5
        
    
    def forward(self, hidden_states):
        B, H, W, _ = hidden_states.shape

        self.video_length = int(B/ self.batch_size)
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(hidden_states).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        q = rearrange(q, "(b n f) (h w) c -> b n f h w c", b=self.batch_size, n=self.num_heads, f=self.video_length, h=H, w=W)
        k = rearrange(k, "(b n f) (h w) c -> b n f h w c", b=self.batch_size, n=self.num_heads, f=self.video_length, h=H, w=W)
        v = rearrange(v, "(b n f) (h w) c -> b n f h w c", b=self.batch_size, n=self.num_heads, f=self.video_length, h=H, w=W)

        q = torch.cat([q[:, :, :1, :, :], q[:, :, 1:, :, :]], dim=2)
        k = torch.cat([k[:, :, -1:, :, :], k[:, :, :-1, :, :]], dim=2)
        v = torch.cat([v[:, :, -1:, :, :], v[:, :, :-1, :, :]], dim=2)

        q = q.reshape(self.batch_size, self.num_heads, self.video_length, H * W, -1)
        k = k.reshape(self.batch_size, self.num_heads, self.video_length, -1, H * W)
        v = v.reshape(self.batch_size, self.num_heads, self.video_length, H * W, -1)

        attn = torch.einsum('bnfhw, bnfwh -> bnhfw', q * self.scale, k)
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bnhfw, bnfhw -> bnhfw', attn, v)
        
        x = x.view(self.batch_size, self.num_heads, self.video_length, H, W, -1)
        x = rearrange(x, "b n f h w c -> (b f) h w (n c)", f=self.video_length)

        return x
