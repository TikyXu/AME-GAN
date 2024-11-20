# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math

from collections import OrderedDict
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from .layers import (
    BatchNorm2d,
    Conv2d,
    FrozenBatchNorm2d,
    interpolate,
    _NewEmptyTensorOp
)
from .fbnet_modeldef import MODEL_ARCH

logger = logging.getLogger(__name__)

def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


PRIMITIVES = {    
    # Transformer Block
    # [dim, heads, dim_head, mlp_dim, dropout]
    "tb_h4_f32": lambda dim: TransformerBlock(
        dim, 4, 32, 32, 0.1
    ),
    "tb_h4_f64": lambda dim: TransformerBlock(
        dim, 4, 32, 64, 0.1
    ),
    "tb_h4_f128": lambda dim: TransformerBlock(
        dim, 4, 32, 128, 0.1
    ),
    "tb_h4_f256": lambda dim: TransformerBlock(
        dim, 4, 32, 256, 0.1
    ),
    "tb_h4_f512": lambda dim: TransformerBlock(
        dim, 4, 32, 512, 0.1
    ),
    "tb_h8_f32": lambda dim: TransformerBlock(
        dim, 8, 16, 32, 0.1
    ),
    "tb_h8_f64": lambda dim: TransformerBlock(
        dim, 8, 16, 64, 0.1
    ),
    "tb_h8_f128": lambda dim: TransformerBlock(
        dim, 8, 16, 128, 0.1
    ),
    "tb_h8_f256": lambda dim: TransformerBlock(
        dim, 8, 16, 256, 0.1
    ),
    "tb_h8_f512": lambda dim: TransformerBlock(
        dim, 8, 16, 512, 0.1
    ),
    "tb_h16_f32": lambda dim: TransformerBlock(
        dim, 16, 8, 32, 0.1
    ),
    "tb_h16_f64": lambda dim: TransformerBlock(
        dim, 16, 8, 64, 0.1
    ),
    "tb_h16_f128": lambda dim: TransformerBlock(
        dim, 16, 8, 128, 0.1
    ),
    "tb_h16_f256": lambda dim: TransformerBlock(
        dim, 16, 8, 256, 0.1
    ),
    "tb_h16_f512": lambda dim: TransformerBlock(
        dim, 16, 8, 512, 0.1
    ),
    "skip_transformer": lambda dim: TransformerIdentity(),
    
}
   
class PositionalEmbedding(nn.Module):
    def __init__(self, *, seq_len, patch_size, dim, channels, emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size
        # patch_dim = patch_size
        self.patch_dim = [patch_size, channels, patch_dim]
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            # batch_size channels (patch_number * patch_size) -> batch_size patch_number (patch_size * channels)
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        
        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        return x, ps


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
                
        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        qkv1 = self.to_qkv1(x).chunk(3, dim = -1)
        _, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv1)

        qkv2 = self.to_qkv2(y).chunk(3, dim = -1)
        q2, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv2)
        
        dots = torch.matmul(q2, k1.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
 
   
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
 
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layernorm1 = nn.modules.LayerNorm(dim)
        self.multiheadselfattention = MultiHeadSelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.layernorm2 = nn.modules.LayerNorm(dim)
        self.feedforward = FeedForward(dim, mlp_dim, dropout = dropout)
        
    def forward(self, x):
        ln1 = self.layernorm1(x)
        mhsa = self.multiheadselfattention(x)
        add1 = ln1 + mhsa
        
        ln2 = self.layernorm2(add1)
        ff = self.feedforward(ln2)
        # add2 = ff + add1
        add2 = ff + add1 + x
        
        return add2


class TransformerBlockWithCrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layernorm1 = nn.modules.LayerNorm(dim)
        self.multiheadcrossattention = MultiHeadCrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.layernorm2 = nn.modules.LayerNorm(dim)
        self.feedforward = FeedForward(dim, mlp_dim, dropout = dropout)
        
    def forward(self, x, y):
        ln1 = self.layernorm1(x)
        mhca = self.multiheadcrossattention(x, y)
        add1 = ln1 + mhca
        
        ln2 = self.layernorm2(add1)
        ff = self.feedforward(ln2)
        add2 = ff + add1
        
        return add2


class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_classes, dim, dropout):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp_head(x)

  
class TransformerIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x