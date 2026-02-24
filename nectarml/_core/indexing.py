from collections.abc import Callable

import numpy as np

def gather(
    input: np.ndarray, 
    dim: int, 
    index: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.take_along_axis(input, index.astype(int), axis=dim)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(input)
        np.add.at(
            grad, 
            tuple(np.arange(s) if i != dim else index.astype(int) 
            for i, s in enumerate(input.shape)), out_grad)
        return grad
    return out, _backward

def scatter(
    input: np.ndarray, 
    dim: int, 
    index: np.ndarray, 
    src: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out_data = input.copy()
    np.put_along_axis(out_data, index.astype(int), src, axis=dim)
    out = out_data
    
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src_grad = np.take_along_axis(out_grad, index.astype(int), axis=dim)

        input_grad = out_grad.copy()
        np.put_along_axis(input_grad, index.astype(int), 0, axis=dim)
        
        return src_grad, input_grad 
    
    return out, _backward

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
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.where(mask, value, input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.where(mask, 0, out_grad)
    return out, _backward

def index_select(
    input: np.ndarray, 
    dim: int, 
    index: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.take(input, index.astype(int), axis=dim)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(input)
        np.add.at(
            grad, 
            tuple(index.astype(int) if i == dim else slice(None) 
            for i in range(input.ndim)), out_grad)
        return grad
    return out, _backward

