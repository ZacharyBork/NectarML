import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

### SUM ###

def sum(
    in_ptr: int, 
    size: int, 
    dtype: DTypeLike, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> int:
    out = _nectarml.reduce_sum(in_ptr, size, map_dtype(dtype))
    return out
