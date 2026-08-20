from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml

def concatenate(
    inputs: list[Tensor],
    dim:    int
) -> int:
    in_ptrs = [i._data_ptr for i in inputs]
    shapes  = [list(i.shape) for i in inputs]
    return _nectarml.tensor.combination.concatenate(
        in_ptrs, shapes, dim, inputs[0].dtype.cuda)

