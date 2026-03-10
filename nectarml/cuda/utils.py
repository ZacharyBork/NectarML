from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

from typing import Any

import numpy as np

import _nectarml
from nectarml.cuda.mapping import DTYPE_MAP
from nectarml.cuda.memory import free_cuda
from nectarml.typing import DTypeLike

class CudaBuffer:
    def __init__(self, ptr: int, dtype: DTypeLike) -> None:
        self.ptr = ptr
        self.dtype = dtype
        self._ref_count = 1
        
    def increment(self) -> CudaBuffer:
        self._ref_count += 1
        return self
        
    def decrement(self) -> None:
        self._ref_count -= 1
        if self._ref_count == 0:
            if free_cuda is not None: free_cuda(self.ptr)

def map_dtype(dtype: DTypeLike) -> Any:
    return DTYPE_MAP[dtype]

def to_cuda(input: Tensor) -> int:
    ptr = _nectarml.to_cuda(
        input.data.ctypes.data, input.size, map_dtype(input.dtype))
    input.data = None
    return ptr

def to_cpu(input: Tensor, host_dtype: DTypeLike | None = None) -> np.ndarray:
    cast_dtype = host_dtype or input.dtype
    data = _nectarml.to_cpu(
        input._data_ptr, list(input.shape), map_dtype(input.dtype))
    return data.astype(cast_dtype)

def cast_tensor(input: Tensor, new_dtype: DTypeLike) -> int:
    return _nectarml.cast_tensor(
        input._data_ptr, input.size, 
        map_dtype(input.dtype), map_dtype(new_dtype))
    
def compute_tensor_min(input: Tensor) -> float:
    return _nectarml.compute_tensor_min(
        input._data_ptr, input.size, map_dtype(input.dtype))

def compute_tensor_max(input: Tensor) -> float:
    return _nectarml.compute_tensor_max(
        input._data_ptr, input.size, map_dtype(input.dtype))

def compute_tensor_range(input: Tensor) -> list[float]:
    return _nectarml.compute_tensor_range(
        input._data_ptr, input.size, map_dtype(input.dtype))


