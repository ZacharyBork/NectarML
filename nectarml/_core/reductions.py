from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

def min(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None, 
    keepdims: bool = False
) -> np.ndarray:
    return input.min(axis=dim, keepdims=keepdims)

def max(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> np.ndarray:
    return input.max(axis=dim, keepdims=keepdims)

def argmin(
    input: np.ndarray,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.argmin(axis=dim, keepdims=keepdims)
    
def argmax(
    input: np.ndarray,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.argmax(axis=dim, keepdims=keepdims)

def mean(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = input.mean(axis=dim, keepdims=keepdims)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        n = input.size if dim is None else input.shape[dim]
        mask = (np.ones_like(input) / n).astype(input.dtype)
        if not keepdims and dim is not None:
            out_grad = np.expand_dims(out_grad, axis=dim)
        return mask * np.broadcast_to(out_grad, input.shape)
    return out, _backward

def sum(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = input.sum(axis=dim, keepdims=keepdims, initial=initial)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        ones = np.ones_like(input).astype(input.dtype)
        if not keepdims and dim is not None:
            out_grad = np.expand_dims(out_grad, axis=dim)
        return ones * np.broadcast_to(out_grad, input.shape)
    return out, _backward

def prod(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = input.prod(axis=dim, keepdims=keepdims, initial=initial)
    def _backward(out_grad):
        out_expanded = out if keepdims or dim is None \
            else np.expand_dims(out, axis=dim)
        if not keepdims and dim is not None:
            out_grad = np.expand_dims(out_grad, axis=dim)
        return np.broadcast_to(out_expanded / input * out_grad, input.shape)
    return out, _backward
