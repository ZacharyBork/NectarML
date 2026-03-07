from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

def equal_mask(a: Tensor, value: float) -> int:
    return _nectarml.equal_mask(
        a._data_ptr, value, a.size, map_dtype(a.dtype))
    
def less_than_mask(a: Tensor, value: float) -> int:
    return _nectarml.less_than_mask(
        a._data_ptr, value, a.size, map_dtype(a.dtype))

def less_than_or_equal_mask(a: Tensor, value: float) -> int:
    return _nectarml.less_than_or_equal_mask(
        a._data_ptr, value, a.size, map_dtype(a.dtype))

def greater_than_mask(a: Tensor, value: float) -> int:
    return _nectarml.greater_than_mask(
        a._data_ptr, value, a.size, map_dtype(a.dtype))

def greater_than_or_equal_mask(a: Tensor, value: float) -> int:
    return _nectarml.greater_than_or_equal_mask(
        a._data_ptr, value, a.size, map_dtype(a.dtype))

