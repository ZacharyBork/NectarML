from __future__ import annotations

from collections.abc import Sequence, Callable

import nectarml.functional as F
from nectarml           import typing
from nectarml.core      import Tensor
from nectarml.nn.module import Module
from nectarml.creation  import ones, zeros

### BATCH ###

class _BatchNorm(Module):
    def __init__(
        self:                _BatchNorm, 
        parameter_shape:     tuple[int, ...],
        norm_dims:           tuple[int, ...],
        norm_func:           Callable,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = True,
        track_running_stats: bool  = True,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__()
        self.norm_dims = norm_dims
        self.norm_func = norm_func
        
        self.eps                 = eps
        self.momentum            = momentum
        self.track_running_stats = track_running_stats
        self.fused               = fused
                
        if affine:
              self.gamma =  ones(parameter_shape, dtype, 'cpu', True)
              self.beta  = zeros(parameter_shape, dtype, 'cpu', True)
        else: self.gamma = self.beta = None
        
        if track_running_stats:
            self.register_buffer(
                'running_mean', zeros(parameter_shape, requires_grad=False),
                persistent=True, pin_dtype=typing.float32)
            self.register_buffer(
                'running_var', ones(parameter_shape, requires_grad=False),
                persistent=True, pin_dtype=typing.float32)
        
    def forward(self: _BatchNorm, x: Tensor) -> Tensor:
        if self.training:
            x_norm, (mean, var) = self.norm_func(
                x, self.gamma, self.beta, self.eps, self.fused)
            
            if self.track_running_stats:
                M        = self.momentum
                mean_f32 = mean.to(dtype=typing.float32)
                var_f32  = var.to(dtype=typing.float32)
                self.running_mean = (
                    (1 - M) * self.running_mean + M * mean_f32).detach()
                self.running_var  = (
                    (1 - M) * self.running_var  + M * var_f32).detach()
        else:
            if self.track_running_stats:
                x_norm = (x - self.running_mean) 
                x_norm = x_norm / (self.running_var + self.eps).sqrt()
            else:
                mean   = x.mean(dim=self.norm_dims, keepdim=True)
                var    = ((x - mean)**2).mean(dim=self.norm_dims, keepdim=True)
                x_norm = (x - mean) / (var + self.eps).sqrt()
            if self.gamma is not None: x_norm = self.gamma * x_norm
            if self.beta  is not None: x_norm = self.beta  + x_norm
        
        return x_norm.to(dtype=x.dtype)

class BatchNorm1d(_BatchNorm):
    def __init__(
        self:                BatchNorm1d, 
        num_features:        int,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = True,
        track_running_stats: bool  = True,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__(
            (1, num_features, 1), (0,), F.batch_norm1d, eps, momentum, affine, 
            track_running_stats, dtype, fused)

class BatchNorm2d(_BatchNorm):
    def __init__(
        self:                BatchNorm2d, 
        num_features:        int,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = True,
        track_running_stats: bool  = True,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1), (0, 2, 3), F.batch_norm2d, eps, momentum, 
            affine, track_running_stats, dtype, fused)

class BatchNorm3d(_BatchNorm):
    def __init__(
        self:                BatchNorm3d, 
        num_features:        int,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = True,
        track_running_stats: bool  = True,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1, 1), (0, 2, 3, 4), F.batch_norm3d, eps, 
            momentum, affine, track_running_stats, dtype, fused)

### INSTANCE ###

class InstanceNorm1d(_BatchNorm):
    def __init__(
        self:                InstanceNorm1d, 
        num_features:        int,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = False,
        track_running_stats: bool  = False,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__(
            (1, num_features, 1), (2,), F.instance_norm1d, eps, 
            momentum, affine, track_running_stats, dtype, fused)
        
class InstanceNorm2d(_BatchNorm):
    def __init__(
        self:                InstanceNorm2d, 
        num_features:        int,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = False,
        track_running_stats: bool  = False,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1), (2, 3), F.instance_norm2d, eps, 
            momentum, affine, track_running_stats, dtype, fused)
        
class InstanceNorm3d(_BatchNorm):
    def __init__(
        self:                InstanceNorm3d, 
        num_features:        int,
        eps:                 float = 0.00001,
        momentum:            float = 0.1,
        affine:              bool  = False,
        track_running_stats: bool  = False,
        dtype:       typing.dtype  = typing.float32,
        fused:               bool  = True
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1, 1), (2, 3, 4), F.instance_norm3d, eps, 
            momentum, affine, track_running_stats, dtype, fused)
        
### GROUP ###
        
class GroupNorm(Module):
    def __init__(
        self:         GroupNorm, 
        num_groups:   int,
        num_channels: int,
        eps:          float = 0.00001,
        affine:       bool  = True,
        dtype: typing.dtype = typing.float32
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.eps        = eps
        
        if affine:
              self.gamma =  ones((1, num_channels, 1, 1), dtype, 'cpu', True)
              self.beta  = zeros((1, num_channels, 1, 1), dtype, 'cpu', True)
        else: self.gamma = self.beta = None
        
    def forward(self: GroupNorm, x: Tensor) -> Tensor:
        x_norm, _ = F.group_norm(
            x, self.num_groups, self.gamma, self.beta, self.eps) 
        return x_norm.to(dtype=x.dtype)
    
### LAYER ###
    
class LayerNorm(Module):
    def __init__(
        self:               LayerNorm, 
        normalized_shape:   Sequence[int],
        eps:                float = 0.00001,
        elementwise_affine: bool  = True,
        bias:               bool  = True,
        dtype:      typing.dtype  = typing.float32
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps              = eps
        
        if elementwise_affine:
            self.gamma = ones(normalized_shape, dtype, 'cpu', True)
            if bias:
                  self.beta = zeros(normalized_shape, dtype, 'cpu', True)
            else: self.beta = None
        else: self.gamma = self.beta = None
        
    def forward(self: LayerNorm, x: Tensor) -> Tensor:
        x_norm = F.layer_norm(
            x, self.normalized_shape, self.gamma, self.beta, self.eps)
        return x_norm.to(dtype=x.dtype)
