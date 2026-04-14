from __future__ import annotations
import builtins

import numpy as np

import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import data_to_cuda, map_dtype

### UTILS ###

def cuda_synchronize() -> None:
    _nectarml.cuda_synchronize()

### STATISTICS ###

def get_cuda_meminfo() -> tuple[builtins.int, builtins.int, builtins.int]:
    return _nectarml.get_cuda_meminfo() # (total, free, used)

def memory_allocated() -> builtins.int:
    _, _, used = get_cuda_meminfo()
    return used/1024**3

def memory_free() -> builtins.int:
    _, free, _ = get_cuda_meminfo()
    return free/1024**3

def get_memory_statistics(precision: builtins.int = 2) -> str:
    stats = get_cuda_meminfo()
    mb = [round(i/1024**3, precision) for i in stats]
    total, free, used = mb
    return f'Total: {total} | Free: {free} | Used: {used}'

### ALLOCATION / DEALLOCATION

def free_cuda(device_ptr: builtins.int) -> None:
    _nectarml.free_cuda(device_ptr)
    
def alloc_cuda_empty_raw(size_bytes: builtins.int) -> builtins.int:
    return _nectarml.alloc_cuda_empty_raw(size_bytes)

def memcpy_to_cuda(dst_ptr: builtins.int, data: np.ndarray) -> None:
    _nectarml.memcpy_to_cuda(dst_ptr, data.ctypes.data, data.nbytes)

def alloc_cuda_full(
    n_elements: builtins.int, 
    dtype:      DTypeLike, 
    fill_value: builtins.float
) -> builtins.int:
    return _nectarml.alloc_cuda_full(n_elements, map_dtype(dtype), fill_value)

def alloc_cuda_random(
    n_elements: builtins.int, 
    dtype:      DTypeLike, 
    seed:       builtins.int = 12345,
    min_value:  builtins.float = 0.0,
    max_value:  builtins.float = 1.0
) -> builtins.int:
    return _nectarml.alloc_cuda_random(
        n_elements, map_dtype(dtype), seed, min_value, max_value)

def alloc_cuda_empty(
    n_elements: builtins.int, 
    dtype:      DTypeLike
) -> builtins.int:
    return _nectarml.alloc_cuda_empty(n_elements, map_dtype(dtype))

### BUFFER ###

class CudaBuffer:
    def __init__(
        self:        CudaBuffer, 
        ptr_or_data: builtins.int | np.ndarray, 
        dtype:       DTypeLike
    ) -> None:
        self.dtype = dtype
        self._ref_count = 1
        
        if not isinstance(ptr_or_data, builtins.int):
            ptr_or_data = CudaBuffer._from_data(ptr_or_data, dtype)
        self.ptr = ptr_or_data
        
    @staticmethod
    def _from_data(data: np.ndarray, dtype: DTypeLike) -> builtins.int:
        return data_to_cuda(data, data.size, dtype)
        
    def increment(self: CudaBuffer) -> CudaBuffer:
        self._ref_count += 1
        return self
            
    def decrement(self: CudaBuffer) -> None:
        if self.ptr == 0: return
        self._ref_count -= 1
        if self._ref_count <= 0:
            free_cuda(self.ptr)
            self.ptr = 0

    def __del__(self: CudaBuffer) -> None:
        if self.ptr != 0 and self._ref_count > 0:
            free_cuda(self.ptr)
            self.ptr = 0

