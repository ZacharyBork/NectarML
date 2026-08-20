from nectarml        import cuda
from nectarml.core   import Tensor
from nectarml.typing import float32

### BATCH ###

def _batch_norm(
    dim:   int | tuple[int, ...],
    x:     Tensor,
    gamma: Tensor | None,
    beta:  Tensor | None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Abstract for all batch/instance norm functions.

    Args:
        dim   : The dimensions over which to normalize inputs.
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    if fused and x.device == 'cuda': 
        return cuda.fused._bn_fused(dim, x, gamma, beta, eps)
    
    input_dtype = x.dtype
    x_f32       = x.to(dtype=float32)
    
    mean     = x_f32.mean(dim=dim, keepdim=True)
    variance = ((x_f32 - mean)**2).mean(dim=dim, keepdim=True)
    x_norm   = (x_f32 - mean) / (variance + eps).sqrt()
    
    C        = x_f32.shape[1]
    ndim     = x_f32.ndim
    reshape  = tuple(C if i == 1 else 1 for i in range(ndim))
    
    if gamma is not None:
        x_norm = gamma.to(dtype=float32).reshape(reshape) * x_norm
    if beta is not None:
        x_norm = beta.to(dtype=float32).reshape(reshape)  + x_norm
        
    return (
        x_norm.to(dtype=input_dtype),
        (mean.reshape(1, C, 1, 1), variance.reshape(1, C, 1, 1))
    )

def batch_norm1d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applied batch normalization to 3-dimensional tensors (B, C, L).

    Computes a single mean and variance value per channel across all B, H, and 
    W values to normalize activations over batch and spatial dimensions for 
    each channel independently.

    BatchNorm also, by default, maintains a running mean and variance during 
    training which is then sampled from during inference. This can be disabled
    by setting `track_running_stats` to False, in which case the mean and 
    variance will be recalculated each time the module is called.

    Reference:
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
        Accelerating deep network training by reducing internal covariate
        shift." In International conference on machine learning, pp. 
        448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

    Args:
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    assert x.ndim == 3, 'BatchNorm1d expects 3D input (B, C, L)'
    return _batch_norm((0,), x, gamma, beta, eps, fused)

def batch_norm2d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applied batch normalization to 4-dimensional tensors (B, C, H, W).

    Computes a single mean and variance value per channel across all B, H, and 
    W values to normalize activations over batch and spatial dimensions for 
    each channel independently.

    BatchNorm also, by default, maintains a running mean and variance during 
    training which is then sampled from during inference. This can be disabled
    by setting `track_running_stats` to False, in which case the mean and 
    variance will be recalculated each time the module is called.

    Reference:
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
        Accelerating deep network training by reducing internal covariate
        shift." In International conference on machine learning, pp. 
        448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

    Args:
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    assert x.ndim == 4, 'BatchNorm2d expects 4D input (B, C, H, W)'
    return _batch_norm((0, 2, 3), x, gamma, beta, eps, fused)

def batch_norm3d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applied batch normalization to 5-dimensional tensors (B, C, D, H, W).

    Computes a single mean and variance value per channel across all B, H, and 
    W values to normalize activations over batch and spatial dimensions for 
    each channel independently.

    BatchNorm also, by default, maintains a running mean and variance during 
    training which is then sampled from during inference. This can be disabled
    by setting `track_running_stats` to False, in which case the mean and 
    variance will be recalculated each time the module is called.

    Reference:
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
        Accelerating deep network training by reducing internal covariate
        shift." In International conference on machine learning, pp. 
        448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

    Args:
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    assert x.ndim == 5, 'BatchNorm3d expects 5D input (B, C, D, H, W)'
    return _batch_norm((0, 2, 3, 4), x, gamma, beta, eps, fused)

### INSTANCE ###

def instance_norm1d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applied instance normalization to 3-dimensional tensors (B, C, L).

    Normalizes activations over spatial dimensions only for each channel. 
    Computes mean and variance across the spatial dimensions for each (B, C)
    pair so that each sample is normalized independently of the rest of the 
    batch.

    Reference:
        Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance 
        normalization: The missing ingredient for fast stylization." arXiv 
        preprint arXiv:1607.08022 (2016). https://arxiv.org/abs/1607.08022

    Reference:
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
        Accelerating deep network training by reducing internal covariate
        shift." In International conference on machine learning, pp. 
        448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

    Args:
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    assert x.ndim == 3, 'InstanceNorm1d expects 3D input (B, C, L)'
    return _batch_norm((2,), x, gamma, beta, eps, fused)

def instance_norm2d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applied instance normalization to 4-dimensional tensors (B, C, H, W).

    Normalizes activations over spatial dimensions only for each channel. 
    Computes mean and variance across the spatial dimensions for each (B, C)
    pair so that each sample is normalized independently of the rest of the 
    batch.

    Reference:
        Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance 
        normalization: The missing ingredient for fast stylization." arXiv 
        preprint arXiv:1607.08022 (2016). https://arxiv.org/abs/1607.08022

    Reference:
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
        Accelerating deep network training by reducing internal covariate
        shift." In International conference on machine learning, pp. 
        448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

    Args:
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    assert x.ndim == 4, 'InstanceNorm2d expects 4D input (B, C, H, W)'
    return _batch_norm((2, 3), x, gamma, beta, eps, fused)

def instance_norm3d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:   float = 0.00001,
    fused:  bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applied instance normalization to 5-dimensional tensors (B, C, D, H, W).

    Normalizes activations over spatial dimensions only for each channel. 
    Computes mean and variance across the spatial dimensions for each (B, C)
    pair so that each sample is normalized independently of the rest of the 
    batch.

    Reference:
        Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance 
        normalization: The missing ingredient for fast stylization." arXiv 
        preprint arXiv:1607.08022 (2016). https://arxiv.org/abs/1607.08022

    Reference:
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: 
        Accelerating deep network training by reducing internal covariate
        shift." In International conference on machine learning, pp. 
        448-456. pmlr, 2015. https://arxiv.org/abs/1502.03167

    Args:
        x     : The input tensor to normalize.
        gamma : Optional weights tensor for the normalization.
        beta  : Optional bias tensor for the normalization.
        eps   : Small epsilon value to avoid zero-division when normalizing.
        fused : If True, normalization will used fused forward and backward
                kernels for CUDA tensors, which is significantly faster.
    '''
    assert x.ndim == 5, 'InstanceNorm3d expects 5D input (B, C, D, H, W)'
    return _batch_norm((2, 3, 4), x, gamma, beta, eps, fused)

### GROUP ###

def group_norm(
    x:          Tensor,
    num_groups: int,
    gamma:      Tensor | None = None,
    beta:       Tensor | None = None,
    eps:        float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    '''Applies group normalization to input tensors.

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
        x          : The input tensor to normalize.
        num_groups : The number of groups to divide input tensors into.
        gamma      : Optional weights tensor for the normalization.
        beta       : Optional bias tensor for the normalization.
        eps        : Small epsilon value to avoid zero-division.
    '''
    input_dtype = x.dtype
    x_f32       = x.to(dtype=float32)

    assert x_f32.ndim == 4, 'GroupNorm expects 4D input (B, C, H, W)'
    B, C, H, W = x_f32.shape
    G = num_groups
    assert C % G == 0, 'Input channels must be evenly divisible by num_groups.'
    
    x_reshaped = x_f32.reshape((B, G, C//G, H, W))
    mean       = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)
    variance   = ((x_reshaped - mean) ** 2).mean(dim=(2, 3, 4), keepdim=True)
    x_norm     = (x_reshaped - mean) / (variance + eps).sqrt()
    x_norm     = x_norm.reshape(x_f32.shape)
    
    if gamma is not None:
        x_norm = gamma.to(dtype=float32).reshape(1, C, 1, 1) * x_norm
    if beta is not None:
        x_norm = beta.to(dtype=float32).reshape(1, C, 1, 1)  + x_norm
        
    return (
        x_norm.to(dtype=input_dtype),
        (mean.reshape(1, C, 1, 1), variance.reshape(1, C, 1, 1))
    )
    
### LAYER ###

def layer_norm(
    x:                Tensor,
    normalized_shape: list[int],
    gamma:            Tensor | None = None,
    beta:             Tensor | None = None,
    eps:              float = 0.0001
) -> Tensor:
    '''Applies layer normlization to input tensors.

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
        x                : The input tensor to normalize.
        normalized_shape : A ShapeType object defining the dimensions over 
                           which to normalize input activations.
        gamma            : Optional weights tensor for the normalization.
        beta             : Optional bias tensor for the normalization.
        eps              : Small epsilon value to avoid zero-division.
    '''
    input_dtype = x.dtype
    x_f32       = x.to(dtype=float32)
    dims        = tuple(range(-len(normalized_shape), 0))
    
    mean     = x_f32.mean(dim=dims, keepdim=True)
    variance = ((x_f32 - mean) ** 2).mean(dim=dims, keepdim=True)
    x_norm   = (x_f32 - mean) / (variance + eps).sqrt()
    
    if gamma is not None: x_norm = gamma.to(dtype=float32) * x_norm
    if beta is not None:  x_norm = beta.to(dtype=float32) + x_norm
    
    return x_norm.to(dtype=input_dtype)
