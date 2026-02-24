import numpy as np

from nectarml import Tensor
from nectarml._core import indexing

def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    out_data, _backward = indexing.gather(input.data, dim, index.data)
    out = input._build_output_tensor(out_data, (input,))
    def _backward_hook():
        if input.requires_grad:
            input.grad += _backward(out.grad)
    out._backward = _backward_hook
    return out

def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    out_data, _backward = indexing.scatter(
        input.data, dim, index.data, src.data)
    out = input._build_output_tensor(out_data, (input,))
    def _backward_hook():
        src_grad, input_grad = _backward(out.grad)
        if src.requires_grad: src.grad += src_grad
        if input.requires_grad: input.grad += input_grad
    out._backward = _backward_hook
    return out

def where(condition: np.ndarray, x: Tensor, y: Tensor) -> Tensor:
    # NOTE: Needs Boolean Tensor support!!
    out_data, _backward = indexing.where(condition, x.data, y.data)
    out = x._build_output_tensor(out_data, (x, y))
    def _backward_hook():
        x_grad, y_grad = _backward(out.grad)
        if x.requires_grad: x.grad += x_grad
        if y.requires_grad: y.grad += y_grad
    out._backward = _backward_hook
    return out

def masked_fill(input: Tensor, mask: np.ndarray, value: float) -> Tensor:
    # NOTE: Needs Boolean Tensor support!!
    out_data, _backward = indexing.masked_fill(input.data, mask, value)
    out = input._build_output_tensor(out_data, (input,))
    def _backward_hook():
        if input.requires_grad:
            input.grad += _backward(out.grad)
    out._backward = _backward_hook
    return out

def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    out_data, _backward = indexing.index_select(input.data, dim, index.data)
    out = input._build_output_tensor(out_data, (input,))
    def _backward_hook():
        if input.requires_grad:
            input.grad += _backward(out.grad)
    out._backward = _backward_hook
    return out
