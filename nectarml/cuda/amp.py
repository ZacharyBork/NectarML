from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml

def unscale_and_check_grad(
    grad:      Tensor,
    inv_scale: float
) -> bool:
    return _nectarml.amp.unscale_and_check_grad(
        grad._data_ptr, inv_scale, grad.size)


