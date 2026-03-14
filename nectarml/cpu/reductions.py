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
) -> np.ndarray:
    return input.mean(axis=dim, keepdims=keepdims)

def sum(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> np.ndarray:
    return input.sum(axis=dim, keepdims=keepdims, initial=initial)

def prod(
    input: np.ndarray, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> np.ndarray:
    return input.prod(axis=dim, keepdims=keepdims, initial=initial)
