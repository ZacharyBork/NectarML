from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins

import _nectarml
from nectarml import typing

def gather(
    input: Tensor,
    dim:   builtins.int | None, 
    index: Tensor
) -> builtins.int:
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.tensor.indexing.gather(
        input._data_ptr, input.shape, 
        index._data_ptr, index.shape, 
        dim, input.dtype.cuda)
    
def scatter(
    input:  Tensor, 
    dim:    builtins.int,
    index:  Tensor,
    source: Tensor | builtins.int | builtins.float
) -> builtins.int:
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.tensor.indexing.scatter(
        input._data_ptr, input.shape, 
        source._data_ptr, source.shape, 
        index._data_ptr, index.shape, 
        dim, input.dtype.cuda)
    
def scatter_add(
    input:  Tensor, 
    dim:    builtins.int,
    index:  Tensor,
    source: Tensor | builtins.int | builtins.float
) -> builtins.int:
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
        
    output = _nectarml.tensor.indexing.scatter_add(
        input._data_ptr, input.shape, 
        source._data_ptr, source.shape, 
        index._data_ptr, index.shape, 
        dim, input.dtype.cuda)
    
    if input_was_uint8: 
        output = _nectarml.cast_tensor(output, input.size, 
            typing.int32.cuda, typing.uint8.cuda)
    
    return output

def slice_tensor(
    input:  Tensor, 
    starts: list[builtins.int],
    counts: list[builtins.int],
    steps:  list[builtins.int]
) -> builtins.int:
    return _nectarml.tensor.indexing.slice(
        input._data_ptr, input.shape, 
        starts, counts, steps, 
        input.dtype.cuda)
    
def index_put(
    input:  Tensor, 
    starts: list[builtins.int],
    counts: list[builtins.int],
    steps:  list[builtins.int],
    source: Tensor
) -> builtins.int:
    return _nectarml.tensor.indexing.index_put(
        input._data_ptr, list(input.shape), source._data_ptr,
        starts, counts, steps, input.dtype.cuda)
