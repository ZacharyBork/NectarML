from typing import Literal
from collections.abc import Sequence, Callable

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.creation import ones, zeros
from nectarml.nn import Module
import nectarml.functional as F

### BATCH ###

class _BatchNorm(Module):
    def __init__(
        self, 
        parameter_shape: tuple[int, ...],
        norm_dims: tuple[int, ...],
        norm_func: Callable,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.norm_dims = norm_dims
        self.norm_func = norm_func
        
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
                
        if affine:
            self.gamma = ones(
                parameter_shape, 
                dtype=dtype, device=device, requires_grad=True)
            self.beta = zeros(
                parameter_shape,
                dtype=dtype, device=device, requires_grad=True)
        else: self.gamma = self.beta = None
        
        if track_running_stats:
            self.register_buffer(
                'running_mean', 
                zeros(parameter_shape, dtype=dtype, device=device))
            self.register_buffer(
                'running_var', 
                ones(parameter_shape, dtype=dtype, device=device))
        
    def forward(self, x: Tensor) -> Tensor:
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
        self, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1), (0,), F.BatchNorm1d, eps, momentum, affine, 
            track_running_stats, device, dtype)

class BatchNorm2d(_BatchNorm):
    def __init__(
        self, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1), (0, 2, 3), F.BatchNorm2d, eps, momentum, 
            affine, track_running_stats, device, dtype)

class BatchNorm3d(_BatchNorm):
    def __init__(
        self, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1, 1), (0, 2, 3, 4), F.BatchNorm3d, eps, 
            momentum, affine, track_running_stats, device, dtype)

### INSTANCE ###

class InstanceNorm1d(_BatchNorm):
    def __init__(
        self, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1), (2,), F.InstanceNorm1d, eps, 
            momentum, affine, track_running_stats, device, dtype)
        
class InstanceNorm2d(_BatchNorm):
    def __init__(
        self, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1), (2, 3), F.InstanceNorm2d, eps, 
            momentum, affine, track_running_stats, device, dtype)
        
class InstanceNorm3d(_BatchNorm):
    def __init__(
        self, 
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            (1, num_features, 1, 1, 1), (2, 3, 4), F.InstanceNorm3d, eps, 
            momentum, affine, track_running_stats, device, dtype)
        
### GROUP ###
        
class GroupNorm(Module):
    def __init__(
        self, 
        num_groups: int,
        num_channels: int,
        eps: float = 0.00001,
        affine: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.num_groups = num_groups
        self.eps = eps
        
        if affine:
            self.gamma = ones(
                (1, num_channels, 1, 1), 
                dtype=dtype, device=device, requires_grad=True)
            self.beta = zeros(
                (1, num_channels, 1, 1),
                dtype=dtype, device=device, requires_grad=True)
        else: self.gamma = self.beta = None
        
    def forward(self, x: Tensor) -> Tensor:
        x_norm, _ = F.GroupNorm(
            x, self.num_groups, self.gamma, self.beta, self.eps) 
        return x_norm
    
### LAYER ###
    
class LayerNorm(Module):
    def __init__(
        self, 
        normalized_shape: Sequence[int],
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu', 
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        if elementwise_affine:
            self.gamma = ones(
                normalized_shape, dtype=dtype, 
                device=device, requires_grad=True)
            if bias:
                self.beta = zeros(
                    normalized_shape, dtype=dtype, 
                    device=device, requires_grad=True)
            else: self.beta = None
        else: self.gamma = self.beta = None
        
    def forward(self, x: Tensor) -> Tensor:
        return F.LayerNorm(
            x, self.normalized_shape, self.gamma, self.beta, self.eps)

