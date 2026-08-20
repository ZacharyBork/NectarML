from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml

def matmul(a: Tensor, b: Tensor) -> int:
    return _nectarml.tensor.matmul(
        a._data_ptr, b._data_ptr, a.shape, b.shape, a.dtype.cuda)
    
