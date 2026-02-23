import numpy as np

from nectarml import Tensor, DTypeLike, ArrayLike
from nectarml.functional.common import _wrapper_base

def min(
    input: Tensor, 
    dim: int | None = None, 
    keepdims: bool = False
) -> Tensor:
    def _grad(x: np.ndarray):
        max_vals = x.min(axis=dim, keepdims=True)
        return (x == max_vals).astype(x.dtype)
    return _wrapper_base(
        input, lambda x: x.min(axis=dim, keepdims=keepdims), 
        _grad, reduce_dim=dim if not keepdims else None)

def max(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> Tensor:
    def _grad(x: np.ndarray):
        max_vals = x.max(axis=dim, keepdims=True)
        return (x == max_vals).astype(x.dtype)
    return _wrapper_base(
        input, lambda x: x.max(axis=dim, keepdims=keepdims), 
        _grad, reduce_dim=dim if not keepdims else None)

def argmin(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.data.argmin(axis=dim, keepdims=keepdims)
    
def argmax(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.data.argmax(axis=dim, keepdims=keepdims)

def mean(
    input: Tensor,
    dim: int | None = None, 
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
) -> Tensor:
    def _grad(x: np.ndarray):
        n = x.size if dim is None else x.shape[dim]
        return (np.ones_like(x) / n)
    return _wrapper_base(
        input, lambda x: x.mean(axis=dim, dtype=dtype, keepdims=keepdims), 
        _grad, reduce_dim=dim if not keepdims else None)
    
def sum(
    input: Tensor,
    dim: int | None = None, 
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> Tensor:
    return _wrapper_base(
        input, lambda x: x.sum(
            axis=dim, dtype=dtype, keepdims=keepdims, initial=initial),
        np.ones_like, reduce_dim=dim if not keepdims else None)

def prod(
    input: Tensor,
    dim: int | None = None, 
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: int | float = 1
) -> Tensor:
    out = input._build_output_tensor(
        input.data.prod(
            axis=dim, dtype=dtype, keepdims=keepdims, initial=initial),
        (input,))
    def _backward():
        if input.requires_grad:
            out_grad = out.grad
            if dim is not None and not keepdims:
                out_grad = np.expand_dims(out_grad, axis=dim)
            out_data = out.data
            if not keepdims:
                out_data = np.expand_dims(out_data, axis=dim) \
                    if dim is not None else out_data
            input.grad += np.broadcast_to(
                out_data / input.data * out_grad, input.data.shape)
    out._backward = _backward
    return out
