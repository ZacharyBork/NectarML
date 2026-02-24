from collections.abc import Sequence

import numpy as np

from nectarml import Tensor
from nectarml._core import combination

def concatenate(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    out_data, _backward = combination.concatenate(
        [t.data for t in inputs], dim=dim)
    out = inputs[0]._build_output_tensor(out_data, tuple(inputs))
    
    def _backward_hook():
        grads = _backward(out.grad)
        for tensor, grad in zip(inputs, grads):
            if tensor.requires_grad:
                tensor.grad += grad
    
    out._backward = _backward_hook
    return out

def cat(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    return concatenate(inputs, dim)

def stack(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    out_data, _backward = combination.stack([t.data for t in inputs], dim=dim)
    out = inputs[0]._build_output_tensor(out_data, tuple(inputs))
    
    def _backward_hook():
        grads = _backward(out.grad)
        for tensor, grad in zip(inputs, grads):
            if tensor.requires_grad:
                tensor.grad += grad
    
    out._backward = _backward_hook
    return out

def unstack(input: Tensor, dim: int = 0) -> list[Tensor]:
    out_data, _backward = combination.unstack(input.data, dim=dim)
    outputs = [input._build_output_tensor(i, (input,)) for i in out_data]
    
    _backward_called = False
    def _backward_hook():
        nonlocal _backward_called
        if _backward_called: return
        _backward_called = True
        
        if input.requires_grad:
            input.grad += _backward([o.grad for o in outputs])
    
    for out in outputs:
        out._backward = _backward_hook
    return outputs

def unbind(input: Tensor, dim: int = 0) -> list[Tensor]:
    return unstack(input, dim)

def split(
    input: Tensor, 
    sizes: int | Sequence[int], 
    dim: int = 0
) -> list[Tensor]:
    out_data, _backward = combination.split(input.data, sizes=sizes, dim=dim)
    outputs = [input._build_output_tensor(i, (input,)) for i in out_data]

    _backward_called = False
    def _backward_hook():
        nonlocal _backward_called
        if _backward_called: return
        _backward_called = True
        if input.requires_grad:
            input.grad += _backward([o.grad for o in outputs])
    
    for out in outputs:
        out._backward = _backward_hook
    return outputs

def chunk(input: Tensor, size: int, dim: int = 0) -> list[Tensor]:
    assert size >= 1
    chunk_size = int(np.ceil(input.shape[dim] / size))
    return split(input, chunk_size, dim)

