import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

### SUM ###

def sum(
    in_ptr: int, 
    shape: tuple[int, ...],
    reduce_dim: int | None,
    dtype: DTypeLike
) -> int:
    return _nectarml.reduce_sum(in_ptr, shape, reduce_dim, map_dtype(dtype))
