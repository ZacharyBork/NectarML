from collections.abc import Sequence, Callable

import numpy as np

def _manipulate_shape(
    input: np.ndarray, 
    new_data: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    old_shape = input.shape
    out = new_data
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad.reshape(old_shape)
    return out, _backward

def reshape(
    input: np.ndarray, 
    shape: tuple[int, ...]
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    return _manipulate_shape(input, input.reshape(shape))

def flatten(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    return _manipulate_shape(input, input.flatten())

def squeeze(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    return _manipulate_shape(input, input.squeeze(axis=dim))
    
def unsqueeze(
    input: np.ndarray, 
    dim: int | tuple[int, ...]
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    return _manipulate_shape(input, np.expand_dims(input, axis=dim))

def transpose(
    input: np.ndarray, 
    dims: Sequence[int] | None
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.transpose(input, axes=dims)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.transpose(out_grad, axes=dims)
    return out, _backward

def swapdims(
    input: np.ndarray, 
    dim1: int, dim2: int
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = input.swapaxes(dim1, dim2)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad.swapaxes(dim1, dim2)
    return out, _backward

def permute(
    input: np.ndarray, 
    dims: Sequence[int] | None
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.permute_dims(input, axes=dims)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        inverse_axes = np.argsort(dims)
        return np.permute_dims(out_grad, axes=inverse_axes)
    return out, _backward
    
def expand(
    input: np.ndarray, 
    shape: tuple[int, ...]
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.broadcast_to(input, shape)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = out_grad
        ndims_added = grad.ndim - input.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        for i, (in_size, out_size) in enumerate(
            zip(input.shape, grad.shape)):
            if in_size == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(input.shape)
    return out, _backward

