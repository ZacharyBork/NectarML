from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

def min(input: Tensor, reduce_dim: int | None) -> int:
    _dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.reduce_min(input._data_ptr, s, _dtype)
    return _nectarml.reduce_min_dim(input._data_ptr, s, reduce_dim, _dtype)

def max(input: Tensor, reduce_dim: int | None) -> int:
    _dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.reduce_max(input._data_ptr, s, _dtype)
    return _nectarml.reduce_max_dim(input._data_ptr, s, reduce_dim, _dtype)

def mean(input: Tensor, reduce_dim: int | None) -> int:
    _dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.reduce_mean(input._data_ptr, s, _dtype)
    return _nectarml.reduce_mean_dim(input._data_ptr, s, reduce_dim, _dtype)

def sum(input: Tensor, reduce_dim: int | None) -> int:
    _dtype = map_dtype(input.dtype)
    s = list(input.shape) if reduce_dim is not None else input.size
    if reduce_dim is None: 
        return _nectarml.reduce_sum(input._data_ptr, s, _dtype)
    return _nectarml.reduce_sum_dim(input._data_ptr, s, reduce_dim, _dtype)
