from collections.abc import Sequence, Iterable

import numpy as np

from nectarml import Tensor

# SHAPE MANIPULATION

def _manipulate_shape(input: Tensor, new_data: np.ndarray) -> Tensor:
    old_shape = input.data.shape
    out = input._build_output_tensor(new_data, (input,))
    def _backward():
        if input.requires_grad:
            input.grad += out.grad.reshape(old_shape)
    out._backward = _backward
    return out

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

# TENSOR COMBINATION

def concatenate() -> Tensor: pass

def cat() -> Tensor: pass

def stack() -> Tensor: pass

def split() -> Tensor: pass

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


