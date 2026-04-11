from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

from typing import Any

import numpy as np

import _nectarml
from nectarml.cuda.mapping import DTYPE_MAP
from nectarml.typing import DTypeLike

### PYTHON-SIDE UTILS ###

def map_dtype(dtype: DTypeLike) -> Any:
    return DTYPE_MAP[dtype]

### CUDA-SIDE UTILS ###

def cast_tensor(input: Tensor, new_dtype: DTypeLike) -> int:
    cast_dtype = map_dtype(new_dtype)
    return _nectarml.cast_tensor(
        input._data_ptr, input.size, 
        map_dtype(input.dtype), cast_dtype)

def to_cuda(input: Tensor) -> int:
    cast_dtype = map_dtype(input.dtype)
    ptr = _nectarml.to_cuda(
        input.data.ctypes.data, input.size, cast_dtype)
    return ptr

def data_to_cuda(data: np.ndarray, size: int, dtype: DTypeLike) -> int:
    cast_dtype = map_dtype(dtype)
    ptr = _nectarml.to_cuda(data.ctypes.data, size, cast_dtype)
    return ptr

def to_cpu(input: Tensor, host_dtype: DTypeLike | None = None) -> np.ndarray:
    cast_dtype = host_dtype or input.dtype
    data = _nectarml.to_cpu(
        input._data_ptr, list(input.shape), map_dtype(input.dtype))
    return data.astype(cast_dtype)

def clone(input: Tensor) -> int:
    return _nectarml.clone(input._data_ptr, input.size, map_dtype(input.dtype))
    
def compute_tensor_min(input: Tensor) -> float:
    return _nectarml.compute_tensor_min(
        input._data_ptr, input.size, map_dtype(input.dtype))

def compute_tensor_max(input: Tensor) -> float:
    return _nectarml.compute_tensor_max(
        input._data_ptr, input.size, map_dtype(input.dtype))

def compute_tensor_range(input: Tensor) -> list[float]:
    return _nectarml.compute_tensor_range(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
### INSPECTION UTILS ###
    
def is_inf(input: Tensor) -> bool:
    return _nectarml.is_inf(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def is_finite(input: Tensor) -> bool:
    return _nectarml.is_finite(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def is_nan(input: Tensor) -> bool:
    return _nectarml.is_nan(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def has_inf(input: Tensor) -> bool:
    return _nectarml.has_inf(
        input._data_ptr, input.size, map_dtype(input.dtype))
    
def has_nan(input: Tensor) -> bool:
    return _nectarml.has_nan(
        input._data_ptr, input.size, map_dtype(input.dtype))


