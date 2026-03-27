from __future__ import annotations

from collections.abc import Sequence, Callable

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.nn.module import Module
from nectarml.creation import ones, zeros
from nectarml.typing import DTypeLike, float32

### BATCH ###

class _BatchNorm(Module):
    def __init__(
        self: _BatchNorm, 
        parameter_shape: tuple[int, ...],
        norm_dims: tuple[int, ...],
        norm_func: Callable,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(dtype)
        self.norm_dims = norm_dims
        self.norm_func = norm_func
        
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
                
        if affine:
            self.gamma = ones(
                parameter_shape, 
                dtype=dtype, device='cpu', requires_grad=True)
            self.beta = zeros(
                parameter_shape,
                dtype=dtype, device='cpu', requires_grad=True)
        else: self.gamma = self.beta = None
        
        if track_running_stats:
            self.register_buffer(
                'running_mean', 
                zeros(parameter_shape, dtype=dtype, device='cpu'))
            self.register_buffer(
                'running_var', 
                ones(parameter_shape, dtype=dtype, device='cpu'))
        
    def forward(self: _BatchNorm, x: Tensor) -> Tensor:
        if self.training:
            x_norm, (mean, var) = self.norm_func(
                x, self.gamma, self.beta, self.eps)
            
            if self.track_running_stats:
                M = self.momentum
                self.running_mean = (M * self.running_mean + (1 - M) * mean)
                self.running_var = (M * self.running_var + (1 - M) * var)   
        else:
            if self.track_running_stats:
                x_norm = (x - self.running_mean) 
                x_norm = x_norm / (self.running_var + self.eps).sqrt()
            else:
                mean = x.mean(dim=self.norm_dims, keepdims=True)
                var = ((x - mean) ** 2).mean(dim=self.norm_dims, keepdims=True)
                x_norm = (x - mean) / (var + self.eps).sqrt()
            if self.gamma is not None: x_norm = self.gamma * x_norm
            if self.beta is not None: x_norm = self.beta + x_norm
        
        return x_norm

class BatchNorm1d(_BatchNorm):
    def __init__(
        self: BatchNorm1d, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1), (0,), F.BatchNorm1d, eps, momentum, affine, 
            track_running_stats, dtype)

class BatchNorm2d(_BatchNorm):
    def __init__(
        self: BatchNorm2d, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1), (0, 2, 3), F.BatchNorm2d, eps, momentum, 
            affine, track_running_stats, dtype)

class BatchNorm3d(_BatchNorm):
    def __init__(
        self: BatchNorm3d, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1, 1), (0, 2, 3, 4), F.BatchNorm3d, eps, 
            momentum, affine, track_running_stats, dtype)

### INSTANCE ###

class InstanceNorm1d(_BatchNorm):
    def __init__(
        self: InstanceNorm1d, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1), (2,), F.InstanceNorm1d, eps, 
            momentum, affine, track_running_stats, dtype)
        
class InstanceNorm2d(_BatchNorm):
    def __init__(
        self: InstanceNorm2d, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1), (2, 3), F.InstanceNorm2d, eps, 
            momentum, affine, track_running_stats, dtype)
        
class InstanceNorm3d(_BatchNorm):
    def __init__(
        self: InstanceNorm3d, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1, 1), (2, 3, 4), F.InstanceNorm3d, eps, 
            momentum, affine, track_running_stats, dtype)
        
### GROUP ###
        
class GroupNorm(Module):
    def __init__(
        self: GroupNorm, 
        num_groups: int,
        num_channels: int,
        eps: float = 0.00001,
        affine: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(dtype)
        self.num_groups = num_groups
        self.eps = eps
        
        if affine:
            self.gamma = ones(
                (1, num_channels, 1, 1), 
                dtype=dtype, device='cpu', requires_grad=True)
            self.beta = zeros(
                (1, num_channels, 1, 1),
                dtype=dtype, device='cpu', requires_grad=True)
        else: self.gamma = self.beta = None
        
    def forward(self: GroupNorm, x: Tensor) -> Tensor:
        x_norm, _ = F.GroupNorm(
            x, self.num_groups, self.gamma, self.beta, self.eps) 
        return x_norm
    
### LAYER ###
    
class LayerNorm(Module):
    def __init__(
        self: LayerNorm, 
        normalized_shape: Sequence[int],
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(dtype)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        if elementwise_affine:
            self.gamma = ones(
                normalized_shape, dtype=dtype, 
                device='cpu', requires_grad=True)
            if bias:
                self.beta = zeros(
                    normalized_shape, dtype=dtype, 
                    device='cpu', requires_grad=True)
            else: self.beta = None
        else: self.gamma = self.beta = None
        
    def forward(self: LayerNorm, x: Tensor) -> Tensor:
        return F.LayerNorm(
            x, self.normalized_shape, self.gamma, self.beta, self.eps)

