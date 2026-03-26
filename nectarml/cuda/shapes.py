from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

def permute(input: Tensor, dims: tuple[int, ...] | None) -> int:
    return _nectarml.tensor.shapes.permute(
        input._data_ptr, input.shape, list(dims), map_dtype(input.dtype))

def expand(input: Tensor, shape: tuple[int, ...]) -> int:
    return _nectarml.tensor.shapes.expand(
        input._data_ptr, input.shape, list(shape), map_dtype(input.dtype))

