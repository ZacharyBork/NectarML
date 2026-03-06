import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

def concatenate(
    in_ptrs: list[int],
    shapes: list[list[int]],
    dim: int,
    dtype: DTypeLike
) -> int:
    return _nectarml.concatenate(in_ptrs, shapes, dim, map_dtype(dtype))
    

