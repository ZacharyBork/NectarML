from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

from typing import Any

import numpy as np

import _nectarml
from nectarml.cuda.mapping import DTYPE_MAP
from nectarml.typing import DTypeLike

def map_dtype(dtype: DTypeLike) -> Any:
    return DTYPE_MAP[dtype]

def cast_tensor(input: Tensor, new_dtype: DTypeLike) -> int:
    return _nectarml.cast_tensor(
        input._data_ptr, input.size, 
        map_dtype(input.dtype), map_dtype(new_dtype))

def to_cuda(input: Tensor) -> int:
    ptr = _nectarml.to_cuda(
        input.data.ctypes.data, input.size, map_dtype(input.dtype))
    return ptr

def data_to_cuda(data: np.ndarray, size: int, dtype: DTypeLike) -> int:
    ptr = _nectarml.to_cuda(data.ctypes.data, size, map_dtype(dtype))
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


