from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import numpy as np

import _nectarml
from nectarml import typing
from nectarml.cuda.utils import map_dtype

def gather(
    input: Tensor,
    dim: int | None, 
    index: Tensor
) -> int:
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.gather(
        input._data_ptr, input.shape, 
        index._data_ptr, index.shape, 
        dim, map_dtype(input.dtype))
    
def scatter(
    input: Tensor, 
    dim: int,
    index: Tensor,
    source: Tensor | int | float
) -> int:
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.scatter(
        input._data_ptr, input.shape, 
        source._data_ptr, source.shape, 
        index._data_ptr, index.shape, 
        dim, map_dtype(input.dtype))
    
def scatter_add(
    input: Tensor, 
    dim: int,
    index: Tensor,
    source: Tensor | int | float
) -> int:
    '''
    NOTE: uint8_t tensors are autoconverted to int32_t due to limitations of
    atomic operations in CUDA. In this case, results will be cast back to, and 
    returned as uint8_t following the scatter_add operation.
    '''
    input_was_uint8 = False
    if input.dtype == typing.uint8: 
        input_was_uint8 = True
        input = input.to(input.device, dtype=typing.int32)
        source = source.to(source.device, dtype=typing.uint32)
                
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    output = _nectarml.scatter_add(
        input._data_ptr, input.shape, 
        source._data_ptr, source.shape, 
        index._data_ptr, index.shape, 
        dim, map_dtype(input.dtype))
    
    if input_was_uint8: 
        output = _nectarml.cast_tensor(output, input.size, 
            map_dtype(typing.int32), map_dtype(typing.uint8))
    
    return output

def where(condition: Tensor, x: Tensor, y: Tensor) -> int:
    mask = condition.to(typing.float32)
    return (x * mask + y * (1 - mask))._data_ptr
