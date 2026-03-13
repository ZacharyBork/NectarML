from nectarml.tensor import Tensor

### BATCH ###

def _BatchNorm(
    dim: int | tuple[int, ...],
    x: Tensor,
    gamma: Tensor | None,
    beta: Tensor | None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    mean = x.mean(dim=dim, keepdim=True)
    variance = ((x - mean)**2).mean(dim=dim, keepdim=True)
    x_norm = (x - mean) / (variance + eps).sqrt()
    if gamma is not None: x_norm = gamma * x_norm
    if not beta is None: x_norm = beta + x_norm
    return x_norm, (mean, variance)

def BatchNorm1d(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 3, 'BatchNorm1d expects 3D input (B, C, L)'
    return _BatchNorm((0,), x, gamma, beta, eps)

def BatchNorm2d(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'BatchNorm2d expects 4D input (B, C, H, W)'
    return _BatchNorm((0, 2, 3), x, gamma, beta, eps)

def BatchNorm3d(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 5, 'BatchNorm3d expects 5D input (B, C, D, H, W)'
    return _BatchNorm((0, 2, 3, 4), x, gamma, beta, eps)

### INSTANCE ###

def InstanceNorm1d(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 3, 'InstanceNorm1d expects 3D input (B, C, L)'
    return _BatchNorm((2,), x, gamma, beta, eps)

def InstanceNorm2d(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'InstanceNorm2d expects 4D input (B, C, H, W)'
    return _BatchNorm((2, 3), x, gamma, beta, eps)

def InstanceNorm3d(
    x: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 5, 'InstanceNorm3d expects 5D input (B, C, D, H, W)'
    return _BatchNorm((2, 3, 4), x, gamma, beta, eps)

### GROUP ###

def GroupNorm(
    x: Tensor,
    num_groups: int,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'GroupNorm expects 4D input (B, C, H, W)'
    B, C, H, W = x.shape
    G = num_groups
    assert C % G == 0, 'Input channels must be evenly divisible by num_groups.'
    
    x_reshaped = x.reshape((B, G, C//G, H, W))
    mean = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)
    variance = ((x_reshaped - mean) ** 2).mean(dim=(2, 3, 4), keepdim=True)
    x_norm = (x_reshaped - mean) / (variance + eps).sqrt()
    x_norm = x_norm.reshape(x.shape)
    if gamma is not None: x_norm = gamma * x_norm
    if beta is not None: x_norm = beta + x_norm
    
    return x_norm, (mean.reshape(1, C, 1, 1), variance.reshape(1, C, 1, 1))
    
### LAYER ###

def LayerNorm(
    x: Tensor,
    normalized_shape: list[int],
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 0.0001
) -> Tensor:
    dims = tuple(range(-len(normalized_shape), 0))
    
    mean = x.mean(dim=dims, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    x_norm = (x - mean) / (variance + eps).sqrt()
    
    if gamma is not None: x_norm = gamma * x_norm
    if beta is not None: x_norm = beta + x_norm
    
    return x_norm

