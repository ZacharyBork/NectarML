from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

from typing import Literal

import _nectarml
from nectarml.cuda.utils import map_dtype

def pad(
    input: Tensor,
    pad_before: list[int],
    pad_after: list[int],
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0
) -> int:
    return _nectarml.pad(
        input._data_ptr, list(input.shape), pad_before, pad_after,
        mode, value, map_dtype(input.dtype))
