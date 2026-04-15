from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import math
import _nectarml

import builtins

def sort(
    input:      Tensor,
    dim:        builtins.int, 
    descending: builtins.bool
) -> tuple[builtins.int, builtins.int]:
    dim      = dim if dim >= 0 else input.ndim + dim
    outer    = builtins.int(math.prod(input.shape[:dim]))
    dim_size = input.shape[dim]
    inner    = builtins.int(math.prod(input.shape[dim+1:]))
    total    = input.numel()
    
    return _nectarml.tensor.sorting.sort(
        input._data_ptr, total, dim_size,
        outer, inner, descending,
        input.dtype.cuda)


