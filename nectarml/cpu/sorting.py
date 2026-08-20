from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor
    
import numpy as np

def sort(
    input: Tensor, 
    dim: int, 
    descending: bool
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(input.data, axis=dim, kind='stable')
    if descending:
        indices = np.flip(indices, axis=dim).copy()
    out_data = np.take_along_axis(input.data, indices, axis=dim)
    return out_data, indices

