from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import math

import _nectarml
from nectarml.cuda.utils import map_dtype

def min(input: Tensor, reduce_dim: int | None) -> int:
    dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.tensor.reductions.reduce_min(
            input._data_ptr, s, dtype)
    return _nectarml.tensor.reductions.reduce_min_dim(
        input._data_ptr, s, reduce_dim, dtype)

def max(input: Tensor, reduce_dim: int | None) -> int:
    dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.tensor.reductions.reduce_max(
            input._data_ptr, s, dtype)
    return _nectarml.tensor.reductions.reduce_max_dim(
        input._data_ptr, s, reduce_dim, dtype)

def mean(input: Tensor, reduce_dim: int | None) -> int:
    dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.tensor.reductions.reduce_mean(
            input._data_ptr, s, dtype)
    return _nectarml.tensor.reductions.reduce_mean_dim(
        input._data_ptr, s, reduce_dim, dtype)

def sum(input: Tensor, reduce_dim: int | None, initial: int | float) -> int:
    dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.tensor.reductions.reduce_sum(
            input._data_ptr, s, initial, dtype)
    return _nectarml.tensor.reductions.reduce_sum_dim(
        input._data_ptr, s, reduce_dim, initial, dtype)

def prod(input: Tensor, reduce_dim: int | None, initial: int | float) -> int:
    dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.tensor.reductions.reduce_prod(
            input._data_ptr, s, initial, dtype)
    return _nectarml.tensor.reductions.reduce_prod_dim(
        input._data_ptr, s, reduce_dim, initial, dtype)

def cumsum(input: Tensor, dim: int) -> int:
    dim      = dim if dim >= 0 else input.ndim + dim
    outer    = int(math.prod(input.shape[:dim]))
    dim_size = input.shape[dim]
    inner    = int(math.prod(input.shape[dim+1:]))
    total    = input.numel()
    return _nectarml.tensor.reductions.reduce_cumsum(
        input._data_ptr, total, dim_size,
        outer, inner,
        map_dtype(input.dtype))
