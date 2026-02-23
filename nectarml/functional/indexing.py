import numpy as np

from nectarml import Tensor

def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    out = input._build_output_tensor(
        np.take_along_axis(input.data, index.data.astype(int), axis=dim),
        (input,))
    def _backward():
        if input.requires_grad:
            grad = np.zeros_like(input.data)
            np.add.at(grad, 
                tuple(np.arange(s) if i != dim else index.data.astype(int) 
                for i, s in enumerate(input.data.shape)), out.grad)
            input.grad += grad
    out._backward = _backward
    return out

def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    out_data = input.data.copy()
    np.put_along_axis(out_data, index.data.astype(int), src.data, axis=dim)
    out = input._build_output_tensor(out_data, (input, src))
    def _backward():
        if src.requires_grad:
            src.grad += np.take_along_axis(
                out.grad, index.data.astype(int), axis=dim)
        if input.requires_grad:
            grad = out.grad.copy()
            np.put_along_axis(grad, index.data.astype(int), 0, axis=dim)
            input.grad += grad
    out._backward = _backward
    return out

def where(condition: np.ndarray, x: Tensor, y: Tensor) -> Tensor:
    # NOTE: Needs Boolean Tensor support!!
    out = x._build_output_tensor(np.where(condition, x.data, y.data), (x, y))
    def _backward():
        if x.requires_grad:
            x.grad += np.where(condition, out.grad, 0)
        if y.requires_grad:
            y.grad += np.where(condition, 0, out.grad)
    out._backward = _backward
    return out

def masked_fill(input: Tensor, mask: np.ndarray, value: float) -> Tensor:
    # NOTE: Needs Boolean Tensor support!!
    out = input._build_output_tensor(
        np.where(mask, value, input.data), (input,))
    def _backward():
        if input.requires_grad:
            input.grad += np.where(mask, 0, out.grad)
    out._backward = _backward
    return out

def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    out = input._build_output_tensor(
        np.take(input.data, index.data.astype(int), axis=dim), (input,))
    def _backward():
        if input.requires_grad:
            grad = np.zeros_like(input.data)
            np.add.at(grad, 
                tuple(index.data.astype(int) if i == dim else slice(None) 
                for i in range(input.data.ndim)), out.grad)
            input.grad += grad
    out._backward = _backward
    return out

