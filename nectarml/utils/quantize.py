import math
from   typing import Any, overload

from nectarml              import typing
from nectarml.core._tensor import Tensor
from nectarml.nn.module    import Module

### UTILS ###

def _quantize_int8(tensor: Tensor) -> Tensor:
    min_value, max_value = tensor.min(), tensor.max()
    scale      = (max_value - min_value) / 255
    zero_point = math.round(-min_value / scale)
    quantized  = ((tensor / (scale + 1e-8)).round() + zero_point).clamp(0, 255)
    return quantized.to(dtype=typing.int8)

def _dequantize_float32(tensor: Tensor) -> Tensor:
    min_value, max_value = tensor.min(), tensor.max()
    scale      = (max_value - min_value) / 255
    zero_point = math.round(-min_value / scale)
    quantized  = ((tensor / (scale + 1e-8)).round() + zero_point).clamp(0, 255)
    return quantized.to(dtype=typing.int8)

### FUNCTIONS ###

@overload
def quantize(input: Tensor, dtype: typing.dtype) -> Tensor: ...
@overload
def quantize(input: Module, dtype: typing.dtype) -> Module: ...
def quantize(input: Any, dtype: typing.dtype) -> Module:
    match dtype:
        case typing.int8(): quantize_fn = _quantize_int8
        case _: raise ValueError(
            f'{dtype} is not a valid quantization DType. '
            'Valid DTypes are: [int8]')
    
    if   isinstance(input, Tensor): return quantize_fn(input)
    elif isinstance(input, Module):
        pass
    else: raise ValueError(f'Quantize input type not valid: {type(input)}')

@overload
def dequantize(input: Tensor, dtype: typing.dtype) -> Tensor: ...
@overload
def dequantize(input: Module, dtype: typing.dtype) -> Module: ...
def dequantize(input: Any, dtype: typing.dtype) -> Module:
    match dtype:
        case typing.float32(): quantize_fn = _quantize_int8
        case _: raise ValueError(
            f'{dtype} is not a valid dequantization DType. '
            'Valid DTypes are: [int8]')
    
    if   isinstance(input, Tensor): return quantize_fn(input)
    elif isinstance(input, Module):
        pass
    else: raise ValueError(f'Quantize input type not valid: {type(input)}')
