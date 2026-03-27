from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import numpy as np
from numpy.typing import ArrayLike

def min(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None, 
    keepdims: bool = False
) -> np.ndarray:
    return input.data.min(axis=dim, keepdims=keepdims)

def max(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> np.ndarray:
    return input.data.max(axis=dim, keepdims=keepdims)

def argmin(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.data.argmin(axis=dim, keepdims=keepdims)
    
def argmax(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return input.data.argmax(axis=dim, keepdims=keepdims)

def mean(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> np.ndarray:
    return input.data.mean(axis=dim, keepdims=keepdims)

def sum(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> np.ndarray:
    return input.data.sum(axis=dim, keepdims=keepdims, initial=initial)

def prod(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> np.ndarray:
    return input.data.prod(axis=dim, keepdims=keepdims, initial=initial)
