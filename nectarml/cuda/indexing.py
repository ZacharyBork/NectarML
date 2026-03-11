from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml import typing
from nectarml.cuda.utils import map_dtype

def gather(
    input: Tensor, 
    indices: Tensor,
    dim: int | None = None
) -> int:
    assert indices.device == input.device, (
        f'Expecting input Tensor and indices Tensor to be on same device, but '
        f'found two devices, {input.device} and {indices.device}')
    assert indices.dtype in [typing.int, typing.int32], \
        f'Indices tensor must have int | int32 dtype'
                
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.gather(
        input._data_ptr, input.shape, 
        indices._data_ptr, indices.shape, 
        dim, map_dtype(input.dtype))



