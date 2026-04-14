from nectarml.tensor import Tensor
from nectarml.typing import float32

### BATCH ###

def _batch_norm(
    dim:   int | tuple[int, ...],
    x:     Tensor,
    gamma: Tensor | None,
    beta:  Tensor | None,
    eps:   float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    input_dtype = x.dtype
    x_f32       = x.to(dtype=float32)
    
    mean     = x_f32.mean(dim=dim, keepdim=True)
    variance = ((x_f32 - mean)**2).mean(dim=dim, keepdim=True)
    x_norm   = (x_f32 - mean) / (variance + eps).sqrt()
    if gamma is not None: x_norm = gamma.to(dtype=float32) * x_norm
    if not beta is None:  x_norm = beta.to(dtype=float32)  + x_norm
    
    C = x_f32.shape[1]
    return (
        x_norm.to(dtype=input_dtype),
        (mean.reshape(1, C, 1, 1), variance.reshape(1, C, 1, 1))
    )

def batch_norm1d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 3, 'BatchNorm1d expects 3D input (B, C, L)'
    return _batch_norm((0,), x, gamma, beta, eps)

def batch_norm2d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'BatchNorm2d expects 4D input (B, C, H, W)'
    return _batch_norm((0, 2, 3), x, gamma, beta, eps)

def batch_norm3d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 5, 'BatchNorm3d expects 5D input (B, C, D, H, W)'
    return _batch_norm((0, 2, 3, 4), x, gamma, beta, eps)

### INSTANCE ###

def instance_norm1d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 3, 'InstanceNorm1d expects 3D input (B, C, L)'
    return _batch_norm((2,), x, gamma, beta, eps)

def instance_norm2d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'InstanceNorm2d expects 4D input (B, C, H, W)'
    return _batch_norm((2, 3), x, gamma, beta, eps)

def instance_norm3d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 5, 'InstanceNorm3d expects 5D input (B, C, D, H, W)'
    return _batch_norm((2, 3, 4), x, gamma, beta, eps)

### GROUP ###

def group_norm(
    x:          Tensor,
    num_groups: int,
    gamma:      Tensor | None = None,
    beta:       Tensor | None = None,
    eps:                float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
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
    
    if gamma is not None: x_norm = gamma.to(dtype=float32) * x_norm
    if beta  is not None: x_norm = beta.to(dtype=float32)  + x_norm
    
    return (
        x_norm.to(dtype=input_dtype),
        (mean.reshape(1, C, 1, 1), variance.reshape(1, C, 1, 1))
    )
    
### LAYER ###

def layer_norm(
    x: Tensor,
    normalized_shape: list[int],
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.0001
) -> Tensor:
    input_dtype = x.dtype
    x_f32       = x.to(dtype=float32)
    dims        = tuple(range(-len(normalized_shape), 0))
    
    mean = x_f32.mean(dim=dims, keepdim=True)
    variance = ((x_f32 - mean) ** 2).mean(dim=dims, keepdim=True)
    x_norm = (x_f32 - mean) / (variance + eps).sqrt()
    
    if gamma is not None: x_norm = gamma.to(dtype=float32) * x_norm
    if beta is not None:  x_norm = beta.to(dtype=float32) + x_norm
    
    return x_norm.to(dtype=input_dtype)
