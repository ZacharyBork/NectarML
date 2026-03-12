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
    indices: Tensor,
    dim: int | None = None
) -> int:
    assert indices.device == input.device, (
        f'Gather expects input Tensor and indices Tensor to be on same '
        f'device, but found two devices, {input.device} and {indices.device}')
    assert indices.dtype in [typing.int, typing.int32], \
        f'Indices tensor must have [int | int32] dtype'
                
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.gather(
        input._data_ptr, input.shape, 
        indices._data_ptr, indices.shape, 
        dim, map_dtype(input.dtype))
    
def scatter(
    input: Tensor, 
    dim: int,
    indices: Tensor,
    source: Tensor | int | float
) -> int:
    from nectarml import Tensor

    if not isinstance(source, Tensor):
        source = Tensor(np.full(input.shape, fill_value=source), 
            dtype=input.dtype, device=input.device)
    
    if not input.device == indices.device or not input.device == source.device:
        _devices = set([input.device, indices.device, source.device])
        raise ValueError(
            f'Scatter expects all Tensors to be on same device, but found '
            f'multiple devices: {list(_devices)}')
    assert input.dtype == source.dtype, \
        f'Input and source must have the same dtype.'
    
    if indices.dtype != typing.int32:
        indices = indices.to(indices.device, dtype=typing.int32)
                
    if dim is None: dim = -1
    dim = dim if dim >= 0 else input.ndim + dim
    return _nectarml.scatter(
        input._data_ptr, input.shape, 
        source._data_ptr, source.shape, 
        indices._data_ptr, indices.shape, 
        dim, map_dtype(input.dtype))
    
def scatter_add(
    input: Tensor, 
    dim: int,
    indices: Tensor,
    source: Tensor | int | float
) -> int:
    '''
    NOTE: uint8_t tensors are autoconverted to int32_t due to limitations of
    atomic operations in CUDA. In this case, results will be cast back to, and 
    returned as uint8_t following the scatter_add operation.
    '''
    from nectarml import Tensor
    
    assert input.shape == indices.shape, \
        f'Shape of index tensor must match shape of input tensor.'
    
    if not isinstance(source, Tensor):
        source = Tensor(
            np.full(indices.shape, fill_value=source, dtype=input.dtype), 
            dtype=input.dtype, device=input.device)
        
    assert input.dtype == source.dtype, \
        f'Input and source must have the same dtype.'
    
    if not input.device == indices.device or not input.device == source.device:
        _devices = set([input.device, indices.device, source.device])
        raise ValueError(
            f'Scatter expects all Tensors to be on same device, but found '
            f'multiple devices: {list(_devices)}')
    
    if indices.dtype != typing.int32:
        indices = indices.to(indices.device, dtype=typing.int32)
                
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
        indices._data_ptr, indices.shape, 
        dim, map_dtype(input.dtype))
    
    if input_was_uint8: 
        output = _nectarml.cast_tensor(output, input.size, 
            map_dtype(typing.int32), map_dtype(typing.uint8))
    
    return output



