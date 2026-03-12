from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

def concatenate(
    inputs: list[Tensor],
    dim: int
) -> int:
    _dtype = inputs[0].dtype
    in_ptrs = [i._data_ptr for i in inputs]
    shapes = [list(i.shape) for i in inputs]
    return _nectarml.concatenate(in_ptrs, shapes, dim, map_dtype(_dtype))

