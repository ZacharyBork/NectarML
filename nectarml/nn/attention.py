from __future__ import annotations

from typing import Literal

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import zeros
from nectarml.typing import DTypeLike, float32
from nectarml.nn.module import Module
from nectarml.nn.linear import Linear

class MultiheadAttention(Module):
    def __init__(
        self, 
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.batch_first = batch_first
        
        self.W_q = Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = Linear(self.kdim, embed_dim, bias=bias)
        self.W_v = Linear(self.vdim, embed_dim, bias=bias)
        self.W_o = Linear(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = zeros((1, 1, embed_dim), requires_grad=True)
            self.bias_v = zeros((1, 1, embed_dim), requires_grad=True)

    def forward(
        self: MultiheadAttention, 
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False
    ) -> Tensor:
        if self.batch_first:
            query = query.transpose(0, 1)
            key   = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        assert self.embed_dim % self.num_heads == 0
        projection_dim = self.embed_dim // self.num_heads

        query = self.W_q(query)
        key   = self.W_k(key)
        value = self.W_v(value)
        
        if self.add_zero_attn:
            zero_k = zeros(
                key.shape[:-2] + (1,) + key.shape[-1:], 
                device=key.device)
            zero_v = zeros(
                value.shape[:-2] + (1,) + value.shape[-1:], 
                device=value.device)
            key = F.cat([key, zero_k], dim=-2)
            value = F.cat([value, zero_v], dim=-2)
        
        if self.add_bias_kv:
            key = F.cat(
                [key, self.bias_k.expand(key.shape[0], -1, -1)], dim=1)
            value = F.cat(
                [value, self.bias_v.expand(value.shape[0], -1, -1)], dim=1)

        attns: list[Tensor] = []
        weights: list[Tensor] = []
        
        for i in range(self.num_heads):
            start = projection_dim * i
            end = projection_dim * (i + 1)
            
            Q_h = query[:, :, start:end]
            K_h = key[:, :, start:end]
            V_h = value[:, :, start:end]
            
            attn_out, weight = F.scaled_dot_product_attention(
                Q_h, K_h, V_h, attn_mask, key_padding_mask, 
                self.dropout, is_causal, self.training)
            
            attns.append(attn_out)
            weights.append(weight)
            
        out = self.W_o(F.cat(attns, dim=-1))
        stacked_weights = F.stack(weights, dim=0)
        
        if self.batch_first: out = out.transpose(0, 1)
        if need_weights:
            if average_attn_weights: attn_weights = stacked_weights.mean(dim=0)
            else: attn_weights = stacked_weights
            return out, attn_weights
        return out, None


