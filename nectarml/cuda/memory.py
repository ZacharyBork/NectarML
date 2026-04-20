from __future__ import annotations
import builtins

import numpy as np

import _nectarml
from nectarml import typing
from nectarml.cuda.utils import data_to_cuda

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

### ALLOCATOR POOL ###

def enable_allocator_pool()  -> None: _nectarml.allocator_pool.enable()
def disable_allocator_pool(release: bool = True) -> None: 
    _nectarml.allocator_pool.disable(release)
def release_allocator_pool() -> None: _nectarml.allocator_pool.release()

### ALLOCATION / DEALLOCATION

def free_cuda(
    device_ptr: builtins.int, 
    n_elements: builtins.int, 
    dtype:      typing.dtype
) -> None:
    _nectarml.free_cuda(device_ptr, n_elements, dtype.cuda)

def memcpy_to_cuda(dst_ptr: builtins.int, data: np.ndarray) -> None:
    _nectarml.memcpy_to_cuda(dst_ptr, data.ctypes.data, data.nbytes)

def alloc_cuda_empty(
    n_elements: builtins.int, 
    dtype:      typing.dtype,
) -> builtins.int:
    return _nectarml.alloc_cuda_empty(n_elements, dtype.cuda)

def alloc_cuda_full(
    n_elements: builtins.int, 
    dtype:      typing.dtype, 
    fill_value: builtins.float
) -> builtins.int:
    return _nectarml.alloc_cuda_full(n_elements, dtype.cuda, fill_value)

def fill(
    device_ptr: builtins.int,
    fill_value: builtins.int,
    n_elements: builtins.int, 
    dtype:      typing.dtype
) -> None:
    _nectarml.fill(device_ptr, fill_value, n_elements, dtype.cuda)

def alloc_cuda_random(
    n_elements: builtins.int, 
    dtype:      typing.dtype, 
    seed:       builtins.int = 12345,
    min_value:  builtins.float = 0.0,
    max_value:  builtins.float = 1.0
) -> builtins.int:
    return _nectarml.alloc_cuda_random(
        n_elements, dtype.cuda, seed, min_value, max_value)

### BUFFER ###

class CudaBuffer:
    _n_bytes: builtins.int = 0
    
    def __init__(
        self:        CudaBuffer, 
        ptr_or_data: builtins.int | np.ndarray, 
        size:        builtins.int,
        dtype:       typing.dtype
    ) -> None:
        super().__setattr__('_n_bytes', size * dtype.itemsize)
        
        self.size        = size
        self.dtype       = dtype
        self._ref_count  = 1
        
        if not isinstance(ptr_or_data, builtins.int):
            ptr_or_data = CudaBuffer._from_data(ptr_or_data, dtype)
        self.ptr = ptr_or_data

    @staticmethod
    def _from_data(data: np.ndarray, dtype: typing.dtype) -> builtins.int:
        return data_to_cuda(data, data.size, dtype)

    def increment(self: CudaBuffer) -> CudaBuffer:
        self._ref_count += 1
        return self
            
    def decrement(self: CudaBuffer) -> None:
        if self.ptr == 0: return
        self._ref_count -= 1
        if self._ref_count <= 0:
            free_cuda(self.ptr, self.size, self.dtype)
            self.ptr = 0

