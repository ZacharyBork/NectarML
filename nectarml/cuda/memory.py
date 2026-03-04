import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

### STATISTICS ###

def get_cuda_meminfo() -> tuple[int, int, int]:
    return _nectarml.get_cuda_meminfo() # (total, free, used)

def memory_allocated() -> int:
    _, _, used = get_cuda_meminfo()
    return used

def get_memory_statistics(precision: int = 2) -> str:
    stats = get_cuda_meminfo()
    mb = [round(i/1024**3, precision) for i in stats]
    total, free, used = mb
    return f'Total: {total} | Free: {free} | Used: {used}'

### ALLOCATION / DEALLOCATION

def free_cuda(device_ptr: int) -> None:
    _nectarml.free_cuda(device_ptr)

def alloc_cuda_full(
    n_elements: int, 
    dtype: DTypeLike, 
    fill_value: float
) -> int:
    return _nectarml.alloc_cuda_full(n_elements, map_dtype(dtype), fill_value)

def alloc_cuda_random(
    n_elements: int, 
    dtype: DTypeLike, 
    seed: int = 12345,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> int:
    return _nectarml.alloc_cuda_random(
        n_elements, map_dtype(dtype), seed, min_value, max_value)

def alloc_cuda_empty(
    n_elements: int, 
    dtype: DTypeLike
) -> int:
    return _nectarml.alloc_cuda_empty(n_elements, map_dtype(dtype))

