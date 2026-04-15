from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins
from typing import Literal

import _nectarml

def pad(
    input:      Tensor,
    pad_before: list[builtins.int],
    pad_after:  list[builtins.int],
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: builtins.float = 0.0
) -> builtins.int:
    return _nectarml.tensor.padding.pad(
        input._data_ptr, list(input.shape), pad_before, pad_after,
        mode, value, input.dtype.cuda)
