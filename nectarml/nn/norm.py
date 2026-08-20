from __future__ import annotations

from collections.abc import Callable

import nectarml.nn.functional as F
from nectarml           import typing
from nectarml.core      import Tensor, creation as T
from nectarml.nn.module import Module

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
        '''Abstract parent of instance and batch norm module classes.'''
        super().__init__()
        self.norm_dims = norm_dims
        self.norm_func = norm_func
        
        self.eps                 = eps
        self.momentum            = momentum
        self.track_running_stats = track_running_stats
        self.fused               = fused
                
        if affine:
              self.gamma =  T.ones(
                  parameter_shape, dtype=dtype, 
                  device='cpu', requires_grad=True)
              self.beta  = T.zeros(
                  parameter_shape, dtype=dtype,
                  device='cpu', requires_grad=True)
        else: self.gamma = self.beta = None
        
        if track_running_stats:
            self.register_buffer(
                'running_mean', T.zeros(parameter_shape, requires_grad=False),
                persistent=True, pin_dtype=typing.float32)
            self.register_buffer(
                'running_var', T.ones(parameter_shape, requires_grad=False),
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
        '''1-dimensional batch normalization module.

        Computes a single mean and variance value per channel across all B, H, 
        and W values to normalize activations over batch and spatial dimensions 
        for each channel independently.

        BatchNorm also, by default, maintains a running mean and variance 
        during training which is then sampled from during inference. This can 
        be disabled by setting `track_running_stats` to False, in which case
        the mean and variance will be recalculated each time the module is 
        called.

        Reference:
            Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
            Accelerating deep network training by reducing internal covariate
            shift." In International conference on machine learning, pp. 
            448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

        Args:
            num_features        : The feature count of the input tensors for 
                                  the module.
            eps                 : Small epsilon value to avoid zero-division
                                  during the normalization calculation.
            momentum            : Affects the rate at which the running mean 
                                  and variance of the module adapt to changes.
                                  Only used if `track_running_stats` is 
                                  enabled.
            affine              : Whether to add learnable gamma and beta 
                                  parameters for the module.
            track_running_stats : If True, the mean and variance of the
                                  normalization will be tracked over time, and
                                  the running stats will then be sampled from
                                  for inference, rather than sampling directly
                                  from the calculated mean and average.
            dtype               : Optional dtype to initialize the modules
                                  gamma and beta parameters with. Only used
                                  when `affine` is enabled.
            fused               : Whether to use fused forward and backward
                                  kernels for normalization of CUDA tensors.
                                  This is significantly faster. 
        '''
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
        '''2-dimensional batch normalization module.

        Computes a single mean and variance value per channel across all B, H, 
        and W values to normalize activations over batch and spatial dimensions 
        for each channel independently.

        BatchNorm also, by default, maintains a running mean and variance 
        during training which is then sampled from during inference. This can 
        be disabled by setting `track_running_stats` to False, in which case
        the mean and variance will be recalculated each time the module is 
        called.

        Reference:
            Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
            Accelerating deep network training by reducing internal covariate
            shift." In International conference on machine learning, pp. 
            448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

        Args:
            num_features        : The feature count of the input tensors for 
                                  the module.
            eps                 : Small epsilon value to avoid zero-division
                                  during the normalization calculation.
            momentum            : Affects the rate at which the running mean 
                                  and variance of the module adapt to changes.
                                  Only used if `track_running_stats` is 
                                  enabled.
            affine              : Whether to add learnable gamma and beta 
                                  parameters for the module.
            track_running_stats : If True, the mean and variance of the
                                  normalization will be tracked over time, and
                                  the running stats will then be sampled from
                                  for inference, rather than sampling directly
                                  from the calculated mean and average.
            dtype               : Optional dtype to initialize the modules
                                  gamma and beta parameters with. Only used
                                  when `affine` is enabled.
            fused               : Whether to use fused forward and backward
                                  kernels for normalization of CUDA tensors.
                                  This is significantly faster. 
        '''
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
        '''3-dimensional batch normalization module.

        Computes a single mean and variance value per channel across all B, H, 
        and W values to normalize activations over batch and spatial dimensions 
        for each channel independently.

        BatchNorm also, by default, maintains a running mean and variance 
        during training which is then sampled from during inference. This can 
        be disabled by setting `track_running_stats` to False, in which case
        the mean and variance will be recalculated each time the module is 
        called.

        Reference:
            Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
            Accelerating deep network training by reducing internal covariate
            shift." In International conference on machine learning, pp. 
            448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

        Args:
            num_features        : The feature count of the input tensors for 
                                  the module.
            eps                 : Small epsilon value to avoid zero-division
                                  during the normalization calculation.
            momentum            : Affects the rate at which the running mean 
                                  and variance of the module adapt to changes.
                                  Only used if `track_running_stats` is 
                                  enabled.
            affine              : Whether to add learnable gamma and beta 
                                  parameters for the module.
            track_running_stats : If True, the mean and variance of the
                                  normalization will be tracked over time, and
                                  the running stats will then be sampled from
                                  for inference, rather than sampling directly
                                  from the calculated mean and average.
            dtype               : Optional dtype to initialize the modules
                                  gamma and beta parameters with. Only used
                                  when `affine` is enabled.
            fused               : Whether to use fused forward and backward
                                  kernels for normalization of CUDA tensors.
                                  This is significantly faster. 
        '''
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
        '''1-dimensional instance normalization module.

        Normalizes activations over spatial dimensions only for each channel. 
        Computes mean and variance across the spatial dimensions for each 
        (B, C) pair so that each sample is normalized independently of the rest 
        of the batch.

        Reference:
            Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance 
            normalization: The missing ingredient for fast stylization." arXiv 
            preprint arXiv:1607.08022 (2016). https://arxiv.org/abs/1607.08022

        Args:
            num_features        : The feature count of the input tensors for 
                                  the module.
            eps                 : Small epsilon value to avoid zero-division
                                  during the normalization calculation.
            momentum            : Affects the rate at which the running mean 
                                  and variance of the module adapt to changes.
                                  Only used if `track_running_stats` is 
                                  enabled.
            affine              : Whether to add learnable gamma and beta 
                                  parameters for the module.
            track_running_stats : If True, the mean and variance of the
                                  normalization will be tracked over time, and
                                  the running stats will then be sampled from
                                  for inference, rather than sampling directly
                                  from the calculated mean and average.
            dtype               : Optional dtype to initialize the modules
                                  gamma and beta parameters with. Only used
                                  when `affine` is enabled.
            fused               : Whether to use fused forward and backward
                                  kernels for normalization of CUDA tensors.
                                  This is significantly faster. 
        '''
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
        '''2-dimensional instance normalization module.

        Normalizes activations over spatial dimensions only for each channel. 
        Computes mean and variance across the spatial dimensions for each 
        (B, C) pair so that each sample is normalized independently of the rest 
        of the batch.

        Reference:
            Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance 
            normalization: The missing ingredient for fast stylization." arXiv 
            preprint arXiv:1607.08022 (2016). https://arxiv.org/abs/1607.08022

        Args:
            num_features        : The feature count of the input tensors for 
                                  the module.
            eps                 : Small epsilon value to avoid zero-division
                                  during the normalization calculation.
            momentum            : Affects the rate at which the running mean 
                                  and variance of the module adapt to changes.
                                  Only used if `track_running_stats` is 
                                  enabled.
            affine              : Whether to add learnable gamma and beta 
                                  parameters for the module.
            track_running_stats : If True, the mean and variance of the
                                  normalization will be tracked over time, and
                                  the running stats will then be sampled from
                                  for inference, rather than sampling directly
                                  from the calculated mean and average.
            dtype               : Optional dtype to initialize the modules
                                  gamma and beta parameters with. Only used
                                  when `affine` is enabled.
            fused               : Whether to use fused forward and backward
                                  kernels for normalization of CUDA tensors.
                                  This is significantly faster. 
        '''
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
        '''3-dimensional instance normalization module.

        Normalizes activations over spatial dimensions only for each channel. 
        Computes mean and variance across the spatial dimensions for each 
        (B, C) pair so that each sample is normalized independently of the rest 
        of the batch.

        Reference:
            Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance 
            normalization: The missing ingredient for fast stylization." arXiv 
            preprint arXiv:1607.08022 (2016). https://arxiv.org/abs/1607.08022

        Args:
            num_features        : The feature count of the input tensors for 
                                  the module.
            eps                 : Small epsilon value to avoid zero-division
                                  during the normalization calculation.
            momentum            : Affects the rate at which the running mean 
                                  and variance of the module adapt to changes.
                                  Only used if `track_running_stats` is 
                                  enabled.
            affine              : Whether to add learnable gamma and beta 
                                  parameters for the module.
            track_running_stats : If True, the mean and variance of the
                                  normalization will be tracked over time, and
                                  the running stats will then be sampled from
                                  for inference, rather than sampling directly
                                  from the calculated mean and average.
            dtype               : Optional dtype to initialize the modules
                                  gamma and beta parameters with. Only used
                                  when `affine` is enabled.
            fused               : Whether to use fused forward and backward
                                  kernels for normalization of CUDA tensors.
                                  This is significantly faster. 
        '''
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
        '''Group normalization module.

        Serves as a sort of middle ground between instance normalization and
        layer normalization. Splits activations into a set number of groups 
        along their channel dimension, and normalizes along the spatial and
        channel dimensions of each group independently. This helps it to avoid
        the issues with small batch sizes when using batch normalization. When 
        `num_groups` is equal to the number of channels in the input tensors,
        GroupNorm behaves like InstanceNorm. With `num_groups`=1, GroupNorm 
        instead behaves like LayerNorm.

        Reference:
            Wu, Yuxin, and Kaiming He. "Group normalization." In Proceedings of 
            the European conference on computer vision (ECCV), pp. 3-19. 2018.
            https://arxiv.org/abs/1803.08494

        Args:
            num_groups   : The number of groups to divide input tensors into.
            num_channels : The number of channels in the input tensors.
            eps          : Small epsilon value to avoid zero-division during 
                           the normalization calculation.
            affine       : Whether to add learnable gamma and beta parameters 
                           for the module.
            dtype        : Optional dtype to initialize the modules gamma and
                           beta parameters with. Only used when `affine` is 
                           enabled.
        '''
        super().__init__()
        self.num_groups = num_groups
        self.eps        = eps
        
        if affine:
              self.gamma =  T.ones(
                  (1, num_channels, 1, 1), dtype=dtype, 
                  device='cpu', requires_grad=True)
              self.beta  = T.zeros(
                  (1, num_channels, 1, 1), dtype=dtype, 
                  device='cpu', requires_grad=True)
        else: self.gamma = self.beta = None
        
    def forward(self: GroupNorm, x: Tensor) -> Tensor:
        x_norm, _ = F.group_norm(
            x, self.num_groups, self.gamma, self.beta, self.eps) 
        return x_norm.to(dtype=x.dtype)
    
### LAYER ###
    
class LayerNorm(Module):
    def __init__(
        self:               LayerNorm, 
        normalized_shape:   typing.ShapeType,
        eps:                float = 0.00001,
        elementwise_affine: bool  = True,
        bias:               bool  = True,
        dtype:      typing.dtype  = typing.float32
    ) -> None:
        '''Layer normalization module.

        Normalizes activations over all non-batch dimensions. So for a tensor
        with shape (B, C, H, W), mean and variance will be computed across the
        C, H, and W dimensions. This causes layer normalization to function the
        same regardless of input batch size, making it highly useful for 
        transformer-based networks and RNNs.

        Reference:
            Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer 
            normalization." arXiv preprint arXiv:1607.06450 (2016).
            https://arxiv.org/abs/1607.06450

        Args:
            normalized_shape   : A ShapeType object defining the dimensions 
                                 over which to normalize input activations.
            eps                : Small epsilon value to avoid zero-division 
                                 during the normalization calculation.
            elementwise_affine : If True, the module will be initialized with
                                 learnable weight and bias (if `bias`=True) 
                                 parameters  
            bias               : If True, a learnable beta parameter will be
                                 added to to the module. Only used if
                                 `elementwise_affine` is enabled.
            dtype              : Optional dtype to initialize the modules gamma 
                                 and beta parameters with. Only used when 
                                 `elementwise_affine` is enabled.
        '''
        super().__init__()
        self.normalized_shape = list(normalized_shape)
        self.eps              = eps
        
        if elementwise_affine:
            self.gamma = T.ones(
                normalized_shape, dtype=dtype, 
                device='cpu', requires_grad=True)
            if bias:
                  self.beta = T.zeros(
                      normalized_shape, dtype=dtype, 
                      device='cpu', requires_grad=True)
            else: self.beta = None
        else: self.gamma = self.beta = None
        
    def forward(self: LayerNorm, x: Tensor) -> Tensor:
        x_norm = F.layer_norm(
            x, self.normalized_shape, self.gamma, self.beta, self.eps)
        return x_norm.to(dtype=x.dtype)
