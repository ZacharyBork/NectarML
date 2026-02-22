from collections.abc import Sequence, Callable

import numpy as np
from numpy.typing import DTypeLike, ArrayLike

from nectarml import Tensor

# ABSTRACTS

def _manipulate_shape(input: Tensor, new_data: np.ndarray) -> Tensor:
    old_shape = input.data.shape
    out = input._build_output_tensor(new_data, (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.reshape(old_shape)
    out._backward = _backward
    return out

def _wrapper_base(
    input: Tensor,
    func: Callable[[np.ndarray], np.ndarray], 
    grad_func: Callable[[np.ndarray], np.ndarray], 
    children: tuple[Tensor, ...] | None = None,
    reduce_dim: int | None = None
) -> Tensor:
    if children is None: children = (input,)
    out = input._build_output_tensor(func(input.data), children)
    def _backward():
        if input.requires_grad:
            out_grad = out.grad
            if reduce_dim is not None and out_grad.ndim < input.data.ndim:
                out_grad = np.expand_dims(out_grad, axis=reduce_dim)
            out_grad = np.broadcast_to(out_grad, input.data.shape)
            input.grad += grad_func(input.data) * out_grad
    out._backward = _backward
    return out

# SHAPE MANIPULATION

def reshape(input: Tensor, shape: Sequence[int]) -> Tensor:
    return _manipulate_shape(input, input.data.reshape(shape))

def flatten(input: Tensor) -> Tensor:
    return _manipulate_shape(input, input.data.flatten())

def squeeze(input: Tensor, dim: int | tuple[int, ...] | None) -> Tensor: 
    return _manipulate_shape(input, input.data.squeeze(axis=dim))
    
def unsqueeze(input: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    return _manipulate_shape(input, np.expand_dims(input.data, axis=dim))

def transpose(input: Tensor, axes: Sequence[int] | None) -> Tensor:
    out = input._build_output_tensor(input.data.transpose(axes), (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.transpose(axes)
    out._backward = _backward
    return out

def swapaxes(input: Tensor, axis1: int, axis2: int) -> Tensor: 
    out = input._build_output_tensor(
        input.data.swapaxes(axis1, axis2), (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.swapaxes(axis1, axis2)
    out._backward = _backward
    return out

def permute(input: Tensor, axes: Sequence[int] | None) -> Tensor:
    out = input._build_output_tensor(
        np.permute_dims(input.data, axes=axes), (input,))
    def _backward():
        if input.requires_grad:
            inverse_axes = np.argsort(axes)
            input.grad += np.permute_dims(out.grad, axes=inverse_axes)
    out._backward = _backward
    return out
    
def expand(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    out = input._build_output_tensor(
        np.broadcast_to(input.data, shape), (input,))
    def _backward():
        if input.requires_grad:
            grad = out.grad
            ndims_added = grad.ndim - input.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, (in_size, out_size) in enumerate(
                zip(input.data.shape, grad.shape)):
                if in_size == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            input.grad += grad.reshape(input.data.shape)
    out._backward = _backward
    return out

def broadcast_to(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return expand(input, shape)

# REDUCTIONS

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

# MATH OPS
    
def exp(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.exp, np.exp)

def log(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.log, lambda x: 1 / x)

def sqrt(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))

def sin(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.sin, np.cos)

def cos(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.cos, lambda x: -np.sin(x))

def tanh(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.tanh, lambda x: 1 - np.tanh(x) ** 2)

def sigmoid(input: Tensor) -> Tensor:
    return (exp(-input) + 1) ** -1

# TENSOR COMBINATION

def concatenate(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    out = inputs[0]._build_output_tensor(
        np.concatenate([t.data for t in inputs], axis=dim), tuple(inputs))
    def _backward():
        sizes = [t.data.shape[dim] for t in inputs]
        split_points = np.cumsum(sizes[:-1])
        grads = np.split(out.grad, split_points, axis=dim)
        for tensor, grad in zip(inputs, grads):
            if tensor.requires_grad:
                tensor.grad += grad
    out._backward = _backward
    return out

def cat(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    return concatenate(inputs, dim)

def stack() -> Tensor: pass

def split(
    input: Tensor, 
    sizes: int | Sequence[int], 
    dim: int = 0
) -> list[Tensor]:
    splits = np.split(input.data, sizes, axis=dim)
    outputs = [input._build_output_tensor(s, (input,)) for s in splits]
    def _backward():
        if input.requires_grad:
            input.grad += np.concatenate([o.grad for o in outputs], axis=dim)
    for out in outputs:
        out._backward = _backward
    return outputs

def chunk() -> Tensor: pass

def unbind() -> Tensor: pass

# INDEXING / SELECTION

def gather() -> Tensor: pass

def scatter() -> Tensor: pass

def where() -> Tensor: pass

def mask_fill() -> Tensor: pass

def index_select() -> Tensor: pass

# PADDING

def pad() -> Tensor: pass


