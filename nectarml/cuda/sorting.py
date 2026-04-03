from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import numpy as np

import _nectarml
from nectarml.cuda.utils import map_dtype

def sort(input: Tensor, dim: int, descending: bool) -> tuple[int, int]:
    dim      = dim if dim >= 0 else input.ndim + dim
    outer    = int(np.prod(input.shape[:dim]))
    dim_size = input.shape[dim]
    inner    = int(np.prod(input.shape[dim+1:]))
    total    = input.numel()
    return _nectarml.tensor.sorting.sort(
        input._data_ptr, total, dim_size,
        outer, inner, descending,
        map_dtype(input.dtype))


