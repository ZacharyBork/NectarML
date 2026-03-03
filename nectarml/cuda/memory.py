import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

def free_cuda(device_ptr: int) -> None:
    _nectarml.free_cuda(device_ptr)
    
def alloc_cuda_full(
    n_elements: int, 
    dtype: DTypeLike, 
    fill_value: float
) -> int:
    return _nectarml.alloc_cuda_full(n_elements, map_dtype(dtype), fill_value)

