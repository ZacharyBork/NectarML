from collections.abc import Callable

import numpy as np

def gather(input: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    return np.take_along_axis(input, index.astype(int), axis=dim)

def scatter(
    input: np.ndarray, 
    dim: int, 
    index: np.ndarray, 
    src: np.ndarray
) -> np.ndarray:
    out = input.copy()
    np.put_along_axis(out, index.astype(int), src, axis=dim)
    return out

def scatter_add(
    input: np.ndarray, 
    dim: int, 
    index: np.ndarray, 
    src: np.ndarray
) -> np.ndarray:
    out = input.copy()
    idx = [slice(None)] * input.ndim
    for i in range(index.shape[dim]):
        idx[dim] = index.take(i, axis=dim)
        np.add.at(out, tuple(idx), src.take(i, axis=dim))
    return out

def where(
    condition: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.where(condition, x, y)
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_grad = np.where(condition, out_grad, 0)
        y_grad = np.where(condition, 0, out_grad)
        return x_grad, y_grad
    return out, _backward

def masked_fill(
    input: np.ndarray, 
    mask: np.ndarray, 
    value: float
) -> np.ndarray:
    return np.where(mask, value, input)

def index_select(input: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    return np.take(input, index.astype(int), axis=dim)

