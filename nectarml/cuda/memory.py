from __future__ import annotations
import builtins

import numpy as np

import _nectarml
from nectarml import typing
from nectarml.cuda.utils import data_to_cuda

### STATISTICS ###

def get_cuda_meminfo() -> tuple[builtins.int, builtins.int, builtins.int]:
    '''Gets info about the current CUDA memory usage.
    
    Returns:
        tuple : A tuple containing (total memory, free memory, used memory), 
                measured in bytes.
    '''
    return _nectarml.get_cuda_meminfo()

def memory_allocated() -> builtins.float:
    '''Gets the total CUDA memory currently allocated.
    
    Returns:
        float : The total CUDA memory currently in use, in gigabytes.
    '''
    _, _, used = get_cuda_meminfo()
    return used/1024**3

def memory_free() -> builtins.float:
    '''Gets the total CUDA memory currently free.
    
    Returns:
        float : The total CUDA memory currently free, in gigabytes.
    '''
    _, free, _ = get_cuda_meminfo()
    return free/1024**3

def get_memory_statistics(precision: builtins.int = 2) -> str:
    '''Returns a string with the total, free, and used CUDA memory statistics.
    
    Args:
        precision : The decimal precision of the memory values.
    
    Returns:
        str : A string with the CUDA memory statistics, measured in gigabytes,
              and formatted like so: 
              'Total: {total} | Free: {free} | Used: {used}'
    '''
    stats = get_cuda_meminfo()
    mb    = [round(i/1024**3, precision) for i in stats]
    total, free, used = mb
    return f'Total: {total} | Free: {free} | Used: {used}'

### ALLOCATOR POOL ###

class AllocatorPool:
    def __init__(
        self:             AllocatorPool, 
        max_vram_percent: builtins.int = 20,
        evict_on_oom:     builtins.bool = True
    ) -> None:
        '''CUDA AllocatorPool.
        
        Thin function wrapper to enable interaction with the CUDA allocator
        pool in the host layer.
        
        NOTE: Should not be instantiated by end user. A CUDA AllocatorPool 
        instance is created by default on package initialization, and can be
        accessed via:

            `nectarml.cuda.allocator_pool`
        '''
        self.max_vram_percent = max_vram_percent
        self.evict_on_oom     = evict_on_oom
    
    def enable(self: AllocatorPool)  -> None: 
        '''Enables the CUDA allocator pool.
        
        NOTE: The CUDA allocator pool is enabled by default. This function is
        only required if you would like to toggle its state manually.
        '''
        _nectarml.allocator_pool.enable()
        
    def disable(self: AllocatorPool, release: bool = True) -> None: 
        '''Disables the CUDA allocator pool.
        
        Args:
            release : If True, the allocator pool will free all unallocated 
                    pointers immediately after being disabled.
        '''
        _nectarml.allocator_pool.disable(release)
        
    def release(self: AllocatorPool) -> None: 
        '''Releases the CUDA allocator pool.
        
        All unallocated ptrs will be freed immediately, and all allocated ptrs
        will be flushed from the pool's register, leaving them to be hard freed
        when a release is triggered from the CudaBuffer that owns the pointer.
        '''
        _nectarml.allocator_pool.release()
        
    def set_vram_limit(
        self:    AllocatorPool, 
        percent: float = 20.0
    ) -> None:
        '''Sets the VRAM limit on the allocator pool.
        
        The pool will only hold memory up to the specified limit, as a 
        percentage of maximum system VRAM. After this limit has been reached,
        calls to the pool to free pointers will result in hard frees, releasing
        the given memory immediately.
        
        Args:
            percent : The percentage (0-100%) of VRAM (as a percentage of 
                      total system VRAM) to allot for the allocator pool.
        '''
        assert 0 <= percent <= 100, \
            'Pool limit must be an integer percentage value between 0 and 100.'
        self.max_vram_percent = percent
        _nectarml.allocator_pool.set_vram_percent(self.max_vram_percent/100.0)

    def set_evict_on_oom(self: AllocatorPool, enabled: bool) -> None:
        '''Sets the allocator pool's evict on OOM state.
        
        If evict on OOM is enabled (enabled=True) and a call is made
        to the pool to allocate memory which results in a CUDA OOM error, the
        pool will automatically attempt to release all unallocated pointers,
        then try the allocation again. If the second attempt fails, the program
        will exit.
        
        If it is disabled, the pool will allow CUDA OOM crashes like usual.
        
        This can help to alleviate crashes caused by transient memory spikes,
        and is enabled by default.
        
        Args:
            enabled : Whether to enabled or disable OOM eviction.
        '''
        self.evict_on_oom = enabled
        _nectarml.allocator_pool.set_evict_on_oom(self.evict_on_oom)

allocator_pool = AllocatorPool()

### ALLOCATION / DEALLOCATION

def free_cuda(
    device_ptr: builtins.int, 
    n_elements: builtins.int, 
    dtype:      typing.dtype
) -> None:
    '''Frees the CUDA memory at a given device pointer.
    
    Args:
        device_ptr : The CUDA device pointer to the memory to free.
        n_elements : The number of elements of the memory to free.
        dtype      : The dtype of the memory to free.
    '''
    _nectarml.free_cuda(device_ptr, n_elements, dtype.cuda)

def memcpy_to_cuda(dst_ptr: builtins.int, data: np.ndarray) -> None:
    '''Copies numpy data in host memory directly to CUDA.
    
    Args:
        dst_ptr : The ptr to the address in device memory to copy the data to.
        data    : The numpy array data to copy to device memory.
    '''
    _nectarml.memcpy_to_cuda(dst_ptr, data.ctypes.data, data.nbytes)

def alloc_cuda_empty(
    n_elements: builtins.int, 
    dtype:      typing.dtype,
) -> builtins.int:
    '''Allocates empty CUDA memory for n_elements of the given DType.
    
    Args:
        n_elements : The number of elements to allocate memory for.
        dtype      : The DType of the data to allocate memory for.
        
    Returns:
        int : Integer ptr to the allocated memory's address on device.
    '''
    return _nectarml.alloc_cuda_empty(n_elements, dtype.cuda)

def alloc_cuda_full(
    n_elements: builtins.int, 
    dtype:      typing.dtype, 
    fill_value: builtins.float
) -> builtins.int:
    '''Allocates memory for n_elements of given DType, filled with given value.
    
    Args:
        n_elements : The number of elements to allocate memory for.
        dtype      : The DType of the data to allocate memory for.
        fill_value : The value to fill the allocated memory with.
        
    Returns:
        int : Integer ptr to the allocated memory's address on device.
    '''
    return _nectarml.alloc_cuda_full(n_elements, dtype.cuda, fill_value)

def fill(
    device_ptr: builtins.int,
    fill_value: builtins.int,
    n_elements: builtins.int, 
    dtype:      typing.dtype
) -> None:
    '''Fills memory at the given address on device with the provided value.
    
    Args:
        device_ptr : The pointer to the address in memory of the data to fill.
        fill_value : The value to fill the memory with.
        n_elements : The number of elements of the data to fill.
        dtype      : The DType of the data to fill.
    '''
    _nectarml.fill(device_ptr, fill_value, n_elements, dtype.cuda)

def alloc_cuda_random(
    n_elements: builtins.int, 
    dtype:      typing.dtype, 
    seed:       builtins.int = 12345,
    min_value:  builtins.float = 0.0,
    max_value:  builtins.float = 1.0
) -> builtins.int:
    '''Allocates memory for n_elements DType and fills with random values.
    
    This does not respect global seeding currently, and has not been thoroughly
    tested. Use `nectarml.random.RNG` or the random functions in 
    `nectarml.creation` instead.
    
    Args:
        n_elements : The number of elements to allocate memory for.
        dtype      : The DType of the data to allocate memory for.
        seed       : The seed to use for random number generation.
        min_value  : The minimum allowable value for the random distribution.
        max_value  : The maximum allowable value for the random distribution.
        
    Returns:
        int : Integer ptr to the allocated memory's address on device.
    '''
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
        '''Utility class which manages the lifecycle of CUDA device pointers.
        
        Not meant to be instantiated directly!
        
        Args:
            ptr_or_data : Either an integer pointer to tensor data in CUDA
                          memory, or a numpy.ndarray whos data should be moved
                          to device.
            size        : The size of the data in number of elements.
            dtype       : The DType of the data.
        '''
        super().__setattr__('_n_bytes', size * dtype.itemsize)
        
        self.size        = size
        self.dtype       = dtype
        self._ref_count  = 1
        
        if not isinstance(ptr_or_data, builtins.int):
            ptr_or_data = CudaBuffer._from_data(ptr_or_data, dtype)
        self.ptr = ptr_or_data

    @staticmethod
    def _from_data(data: np.ndarray, dtype: typing.dtype) -> builtins.int:
        '''Moves numpy data to CUDA, returns resulting pointer.
        
        Args:
            data  : The numpy data to move to CUDA.
            dtype : The DType of the data.
            
        Returns:
            int : An integer pointer to the data's address in device memory.
        '''
        return data_to_cuda(data, data.size, dtype)

    def increment(self: CudaBuffer) -> CudaBuffer:
        '''Increments the CudaBuffer's reference count.
        
        Returns:
            CudaBuffer : A reference to the CudaBuffer this function was called
                         from. Useful for things like:
                         
                         new._buffer = other._buffer.increment()
        '''
        self._ref_count += 1
        return self
            
    def decrement(self: CudaBuffer) -> None:
        '''Decrements the CudaBuffer's reference count.
        
        If the decrement results in the CudaBuffer's ref count reaching 0,
        the CUDA memory associated with the buffer will be freed.
        '''
        if self.ptr == 0: return
        self._ref_count -= 1
        if self._ref_count <= 0:
            free_cuda(self.ptr, self.size, self.dtype)
            self.ptr = 0

