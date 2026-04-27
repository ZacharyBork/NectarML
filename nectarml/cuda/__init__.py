### IMPORT SUBMODULES ###

import _nectarml
from . import amp, memory, utils

### RAISE TOP-LEVEL IMPORTS ###

from .tensor import (
    combination, conv, indexing, interpolation, masking, math, matmul,
    padding, pooling, reductions, shapes, sorting
)

from .utils import (
    is_cuda_available, get_cuda_info, cuda_synchronize, inspect_cuda_data,
    is_inf, is_finite, is_nan, has_inf, has_nan, cast_tensor, cast_ptr, 
    to_cuda, data_to_cuda, to_cpu, ptr_to_cpu, clone, clone_ptr
)

from .memory import (
    allocator_pool, free_cuda, alloc_cuda_full, alloc_cuda_random, 
    get_cuda_meminfo, memory_allocated, get_memory_statistics
)

### INITIALIZE CUDA BACKEND ###

CUDA_INFO = get_cuda_info()

if CUDA_INFO.available:
    import atexit
    atexit.register(_nectarml.destroy_cublas_handle)
    allocator_pool.enable()


