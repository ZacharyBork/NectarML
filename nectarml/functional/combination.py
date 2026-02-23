from collections.abc import Sequence

import numpy as np

from nectarml import Tensor

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

def stack(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    out = inputs[0]._build_output_tensor(
        np.stack([t.data for t in inputs], axis=dim), tuple(inputs))
    def _backward():
        grads = np.split(out.grad, len(inputs), axis=dim)
        for tensor, grad in zip(inputs, grads):
            if tensor.requires_grad:
                tensor.grad += np.squeeze(grad, axis=dim)
    out._backward = _backward
    return out

def unstack(input: Tensor, dim: int = 0) -> list[Tensor]:
    _split = np.split(input.data, input.data.shape[dim], axis=dim)
    splits = [np.squeeze(s, axis=dim) for s in _split]
    outputs = [input._build_output_tensor(s, (input,)) for s in splits]
    _backward_called = False
    def _backward():
        nonlocal _backward_called
        if _backward_called: return
        _backward_called = True
        if input.requires_grad:
            input.grad += np.stack([o.grad for o in outputs], axis=dim)
    for out in outputs:
        out._backward = _backward
    return outputs

def unbind(input: Tensor, dim: int = 0) -> list[Tensor]:
    return unstack(input, dim)

def split(
    input: Tensor, 
    sizes: int | Sequence[int], 
    dim: int = 0
) -> list[Tensor]:
    splits = np.split(input.data, sizes, axis=dim)
    outputs = [input._build_output_tensor(s, (input,)) for s in splits]
    _backward_called = False
    def _backward():
        nonlocal _backward_called
        if _backward_called: return
        _backward_called = True
        if input.requires_grad:
            input.grad += np.concatenate([o.grad for o in outputs], axis=dim)
    for out in outputs:
        out._backward = _backward
    return outputs

def chunk(input: Tensor, size: int, dim: int = 0) -> list[Tensor]:
    assert size >= 1
    chunk_size = int(np.ceil(input.shape[dim] / size))
    return split(input, chunk_size, dim)

