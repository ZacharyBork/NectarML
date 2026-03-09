from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

def eq_mask(a: Tensor, b: Tensor | float) -> int:
    _dtype = map_dtype(a.dtype)
    if isinstance(b, Tensor):
        return _nectarml.eq_mask_tensor(
            a._data_ptr, b._data_ptr, a.size, _dtype)
    return _nectarml.eq_mask_scalar(a._data_ptr, b, a.size, _dtype)
    
def lt_mask(a: Tensor, b: Tensor | float) -> int:
    _dtype = map_dtype(a.dtype)
    if isinstance(b, Tensor):
        return _nectarml.lt_mask_tensor(
            a._data_ptr, b._data_ptr, a.size, _dtype)
    return _nectarml.lt_mask_scalar(a._data_ptr, b, a.size, _dtype)

def le_mask(a: Tensor, b: Tensor | float) -> int:
    _dtype = map_dtype(a.dtype)
    if isinstance(b, Tensor):
        return _nectarml.le_mask_tensor(
            a._data_ptr, b._data_ptr, a.size, _dtype)
    return _nectarml.le_mask_scalar(a._data_ptr, b, a.size, _dtype)

def gt_mask(a: Tensor, b: Tensor | float) -> int:
    _dtype = map_dtype(a.dtype)
    if isinstance(b, Tensor):
        return _nectarml.gt_mask_tensor(
            a._data_ptr, b._data_ptr, a.size, _dtype)
    return _nectarml.gt_mask_scalar(a._data_ptr, b, a.size, _dtype)

def ge_mask(a: Tensor, b: Tensor | float) -> int:
    _dtype = map_dtype(a.dtype)
    if isinstance(b, Tensor):
        return _nectarml.ge_mask_tensor(
            a._data_ptr, b._data_ptr, a.size, _dtype)
    return _nectarml.ge_mask_scalar(a._data_ptr, b, a.size, _dtype)

