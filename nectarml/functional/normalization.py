import _nectarml
from nectarml        import typing, cuda
from nectarml.core   import Tensor
from nectarml.typing import float32

### BATCH ###

def _bn_fused(
    dim:   int | tuple[int, ...],
    x:     Tensor,
    gamma: Tensor | None,
    beta:  Tensor | None,
    eps:   float = 0.00001
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    input_dtype = x.dtype
    x_f32       = x._as_fp32()
    
    gamma_f32 = gamma._as_fp32() if gamma is not None else None
    beta_f32  =  beta._as_fp32() if beta  is not None else None
    
    N = x_f32.shape[0]
    C = x_f32.shape[1]
    H = x_f32.shape[2] if x_f32.ndim > 2 else 1
    W = x_f32.shape[3] if x_f32.ndim > 3 else 1
    
    dim_set  = set(dim) if isinstance(dim, (tuple, list)) else {dim}
    reduce_N = 0 in dim_set
    reduce_H = 2 in dim_set if x_f32.ndim > 2 else False
    reduce_W = 3 in dim_set if x_f32.ndim > 3 else False
    
    n_stats  = C if reduce_N else N * C
    mean_ptr = cuda.memory.alloc_cuda_empty(n_stats, typing.float32)
    var_ptr  = cuda.memory.alloc_cuda_empty(n_stats, typing.float32)
    
    out_ptr = _nectarml.tensor.norm.batch_norm_forward(
        x_f32._data_ptr,
        gamma_f32._data_ptr if gamma_f32 is not None else 0,
        beta_f32._data_ptr  if beta_f32  is not None else 0,
        mean_ptr, var_ptr, N, C, H, W,
        int(reduce_N), int(reduce_H), int(reduce_W), eps)
    
    shape = (1, C, 1, 1) if reduce_N else (N, C, 1, 1)
    mean = Tensor._new(mean_ptr, shape, typing.float32, x.device)
    var  = Tensor._new(var_ptr,  shape, typing.float32, x.device)
    
    x_requires_grad     = x.requires_grad
    gamma_requires_grad = gamma is not None and gamma.requires_grad
    beta_requires_grad  = beta  is not None and beta.requires_grad
    any_requires_grad   = x_requires_grad \
                       or gamma_requires_grad \
                       or beta_requires_grad
    
    children = tuple(t for t in [x, gamma, beta] if t is not None)
    out_f32  = Tensor._new(
        out_ptr, x_f32.shape, typing.float32, x.device,
        any_requires_grad, _children=children)
    
    def _backward() -> None:
        dx_ptr     = cuda.memory.alloc_cuda_empty(x_f32.size, typing.float32)\
                  if x_requires_grad else 0
        dgamma_ptr = cuda.memory.alloc_cuda_empty(C, typing.float32) \
                  if gamma_requires_grad else 0
        dbeta_ptr  = cuda.memory.alloc_cuda_empty(C, typing.float32) \
                  if beta_requires_grad  else 0
        
        _nectarml.tensor.norm.batch_norm_backward(
            out_f32.grad._data_ptr, x_f32._data_ptr,
            mean._data_ptr, var._data_ptr, 
            gamma._data_ptr if gamma is not None else 0,
            dx_ptr, dgamma_ptr, dbeta_ptr, N, C, H, W,
            int(reduce_N), int(reduce_H), int(reduce_W), eps)
        
        if x_requires_grad and dx_ptr:
            dx = Tensor._new(dx_ptr, x_f32.shape, typing.float32, x.device)
            target = x if input_dtype == typing.float32 else x
            if target.grad is None: 
                  target.grad  = dx
            else: target.grad += dx
        
        if gamma_requires_grad and dgamma_ptr:
            dg = Tensor._new(dgamma_ptr, gamma.shape, typing.float32, x.device)
            if gamma.grad is None: 
                  gamma.grad  = dg
            else: gamma.grad += dg
        
        if beta_requires_grad and dbeta_ptr:
            db = Tensor._new(dbeta_ptr, beta.shape, typing.float32, x.device)
            if beta.grad is None: 
                  beta.grad  = db
            else: beta.grad += db
    
    out = out_f32.to(dtype=input_dtype) \
       if input_dtype != typing.float32 else out_f32
    out_f32._backward = _backward
    return out, (mean, var)

def _batch_norm(
    dim:   int | tuple[int, ...],
    x:     Tensor,
    gamma: Tensor | None,
    beta:  Tensor | None,
    eps:   float = 0.00001,
    fused: bool  = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    if fused and x.device == 'cuda': 
        return _bn_fused(dim, x, gamma, beta, eps)
    
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
    eps:           float = 0.00001,
    fused:          bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 3, 'BatchNorm1d expects 3D input (B, C, L)'
    return _batch_norm((0,), x, gamma, beta, eps, fused)

def batch_norm2d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001,
    fused:          bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'BatchNorm2d expects 4D input (B, C, H, W)'
    return _batch_norm((0, 2, 3), x, gamma, beta, eps, fused)

def batch_norm3d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001,
    fused:          bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 5, 'BatchNorm3d expects 5D input (B, C, D, H, W)'
    return _batch_norm((0, 2, 3, 4), x, gamma, beta, eps, fused)

### INSTANCE ###

def instance_norm1d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001,
    fused:          bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 3, 'InstanceNorm1d expects 3D input (B, C, L)'
    return _batch_norm((2,), x, gamma, beta, eps, fused)

def instance_norm2d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001,
    fused:          bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 4, 'InstanceNorm2d expects 4D input (B, C, H, W)'
    return _batch_norm((2, 3), x, gamma, beta, eps, fused)

def instance_norm3d(
    x:     Tensor,
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.00001,
    fused:          bool = True
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    assert x.ndim == 5, 'InstanceNorm3d expects 5D input (B, C, D, H, W)'
    return _batch_norm((2, 3, 4), x, gamma, beta, eps, fused)

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
    x: Tensor,
    normalized_shape: list[int],
    gamma: Tensor | None = None,
    beta:  Tensor | None = None,
    eps:           float = 0.0001
) -> Tensor:
    input_dtype = x.dtype
    x_f32       = x.to(dtype=float32)
    dims        = tuple(range(-len(normalized_shape), 0))
    
    mean     = x_f32.mean(dim=dims, keepdim=True)
    variance = ((x_f32 - mean) ** 2).mean(dim=dims, keepdim=True)
    x_norm   = (x_f32 - mean) / (variance + eps).sqrt()
    
    if gamma is not None: x_norm = gamma.to(dtype=float32) * x_norm
    if beta is not None:  x_norm = beta.to(dtype=float32) + x_norm
    
    return x_norm.to(dtype=input_dtype)
