from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml

def unscale_and_check_grad(
    grad:      Tensor,
    inv_scale: float
) -> bool:
    '''Fused unscale/grad check for the AMP GradScaler.
    
    Args:
        grad      : The grad tensor to unscale and check.
        inv_scale : 1.0 / GradScaler scale factor.
        
    Returns:
        bool : True if all values are finite numbers, otherwise False.
    '''
    return _nectarml.amp.unscale_and_check_grad(
        grad._data_ptr, inv_scale, grad.size)


