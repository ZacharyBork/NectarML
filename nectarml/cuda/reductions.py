import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

def min(
    in_ptr: int, 
    size: int | tuple[int, ...],
    reduce_dim: int | None,
    dtype: DTypeLike
) -> int:
    _dtype = map_dtype(dtype)
    if reduce_dim is None: 
        return _nectarml.reduce_min(in_ptr, size, _dtype)
    return _nectarml.reduce_min_dim(in_ptr, size, reduce_dim, _dtype)

def max(
    in_ptr: int, 
    size: int | tuple[int, ...],
    reduce_dim: int | None,
    dtype: DTypeLike
) -> int:
    _dtype = map_dtype(dtype)
    if reduce_dim is None: 
        return _nectarml.reduce_max(in_ptr, size, _dtype)
    return _nectarml.reduce_max_dim(in_ptr, size, reduce_dim, _dtype)

def sum(
    in_ptr: int, 
    size: int | tuple[int, ...],
    reduce_dim: int | None,
    dtype: DTypeLike
) -> int:
    _dtype = map_dtype(dtype)
    if reduce_dim is None: 
        return _nectarml.reduce_sum(in_ptr, size, _dtype)
    return _nectarml.reduce_sum_dim(in_ptr, size, reduce_dim, _dtype)
