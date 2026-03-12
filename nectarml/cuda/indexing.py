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


def slice_tensor(input: Tensor, indices: slice | tuple[slice]) -> int:
    if not isinstance(indices, tuple): indices = (indices,)
    _indices = list(indices)
    while len(_indices) < input.ndim: _indices.append(slice(None))
        
    starts, stops, steps = [], [], []
    for i, idx in enumerate(_indices):
        dim_size = input.shape[i]
        
        start = idx.start
        stop = idx.stop
        step = idx.step if idx.step is not None else 1
        
        if start is None: start = 0
        elif start < 0: start = dim_size + start
        start = max(0, min(dim_size, start))
        
        if stop is None: stop = dim_size
        elif stop < 0: stop = dim_size + stop
        stop = max(0, min(dim_size, stop))
        
        starts.append(start)
        stops.append(stop)
        steps.append(step)
    
    return _nectarml.slice(
        input._data_ptr, input.shape, 
        starts, stops, steps, 
        map_dtype(input.dtype))
    
def index_put(
    input: Tensor, 
    indices: slice | tuple[slice], 
    source: Tensor
) -> int:
    from nectarml import Tensor
    
    if not isinstance(indices, tuple): indices = (indices,)
    _indices = list(indices)
    while len(_indices) < input.ndim: _indices.append(slice(None))
    
    starts, stops, steps = [], [], []
    src_shape = []
    for i, idx in enumerate(indices):
        dim_size = input.shape[i]
        
        if isinstance(idx, int):
            if idx < 0: idx = dim_size + idx
            starts.append(idx)
            stops.append(idx + 1)
            steps.append(1)
        elif isinstance(idx, slice):
            step = idx.step if idx.step is not None else 1
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else dim_size
            
            if start < 0: start = dim_size + start
            if stop < 0: stop = dim_size + stop
            
            start = max(0, min(dim_size, start))
            stop = max(0, min(dim_size, stop))
            starts.append(start)
            stops.append(stop)
            steps.append(step)
            src_shape.append((stop - start + step - 1) // step)
    
    if not isinstance(source, Tensor):
        source = Tensor(
            np.full(src_shape, fill_value=source, dtype=input.dtype),
            dtype=input.dtype, device=input.device)
    elif source.device != input.device:
        source = source.to(input.device)
    
    return _nectarml.index_put(
        input._data_ptr, list(input.shape),
        source._data_ptr, list(source.shape),
        starts, stops, steps,
        map_dtype(input.dtype))


