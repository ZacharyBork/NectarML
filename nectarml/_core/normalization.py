from collections.abc import Sequence, Callable

import numpy as np

y = np.array([1, 2, 3, 3])
z = np.mean(y)

### BATCH ###

def _BatchNorm(
    dim: int | tuple[int, ...],
    x: np.ndarray,
    gamma: np.ndarray | None,
    beta: np.ndarray | None,
    eps: float = 0.00001
) -> tuple[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
        Callable[[np.ndarray], np.ndarray]]:
    mean = np.mean(x, axis=dim, keepdims=True, dtype=x.dtype)
    variance = np.mean((x - mean)**2, axis=dim, keepdims=True, dtype=x.dtype)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    if gamma is not None: x_norm = gamma * x_norm
    if beta is not None: x_norm = beta + x_norm
    
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        if gamma is not None: d_gamma = np.sum(out_grad * x_norm, axis=dim)
        if beta is not None: d_beta = np.sum(out_grad, axis=dim)
        
        # x_norm = gamma * x_norm + beta
        # 2 * x_norm + 1
        # 
        d_xnorm = 
    
    return (x_norm, (mean, variance)), _backward()

### GROUP ###

def GroupNorm(
    x: np.ndarray,
    num_groups: int,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 0.00001
) -> tuple[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
        Callable[[np.ndarray], np.ndarray]]:
    assert x.ndim == 4, 'GroupNorm expects 4D input (B, C, H, W)'
    B, C, H, W = x.shape
    G = num_groups
    assert C % G == 0, 'Input channels must be evenly divisible by num_groups.'
    
    x_reshaped = x.reshape((B, G, C//G, H, W))
    mean = x_reshaped.mean(dim=(2, 3, 4), keepdims=True)
    variance = ((x_reshaped - mean) ** 2).mean(dim=(2, 3, 4), keepdims=True)
    x_norm = (x_reshaped - mean) / (variance + eps).sqrt()
    x_norm = x_norm.reshape(x.shape)
    if gamma is not None: x_norm = gamma * x_norm
    if beta is not None: x_norm = beta + x_norm
    
    return x_norm, (mean.reshape(1, C, 1, 1), variance.reshape(1, C, 1, 1))
    
### LAYER ###

def LayerNorm(
    x: np.ndarray,
    normalized_shape: Sequence[int],
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 0.0001
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    dims = tuple(range(-len(normalized_shape), 0))
    
    mean = x.mean(dim=dims, keepdims=True)
    variance = ((x - mean) ** 2).mean(dim=dims, keepdims=True)
    x_norm = (x - mean) / (variance + eps).sqrt()
    
    if gamma is not None: x_norm = gamma * x_norm
    if beta is not None: x_norm = beta + x_norm
    
    return x_norm

