from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype
from nectarml.constants import FLOAT_MIN, FLOAT_MAX

### COMPARISON ###

def equal(
    a: Tensor, 
    b: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.equal_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))

def less_than(
    a: Tensor, 
    b: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.less_than_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.less_than(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))

def less_than_or_equal(
    a: Tensor, 
    b: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.less_than_or_equal_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.less_than_or_equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))

def greater_than(
    a: Tensor, 
    b: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.greater_than_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.greater_than(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))

def greater_than_or_equal(
    a: Tensor,
    b: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.greater_than_or_equal_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.greater_than_or_equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))

### BASE ###

def add(
    a: Tensor, 
    b: Tensor | int | float, 
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.add_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.add(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))
    

def subtract(
    a: Tensor, 
    b: Tensor | int | float, 
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.subtract_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.subtract(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))
    

def multiply(
    a: Tensor, 
    b: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.multiply_ts(
         a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.multiply(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))
    
def divide(
    a: Tensor, 
    b: Tensor | int | float, 
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(b, int | float):
        return _nectarml.tensor.elementwise.divide_ts(
            a._data_ptr, b, a.size, map_dtype(a.dtype))
    return _nectarml.tensor.elementwise.divide(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        map_dtype(a.dtype))
    

def negate(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.negate(
        x._data_ptr, x.size, map_dtype(x.dtype))

### SQRT ###

def sqrt(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.sqrt(
        x._data_ptr, x.size, map_dtype(x.dtype))

def rsqrt(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.rsqrt(
        x._data_ptr, x.size, map_dtype(x.dtype))

### EXPONENT ###

def exp(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.exp(
        x._data_ptr, x.size, map_dtype(x.dtype))

### LOG ###

def log(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.log(
        x._data_ptr, x.size, map_dtype(x.dtype))

def log2(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.log2(
        x._data_ptr, x.size, map_dtype(x.dtype))

def log10(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.log10(
        x._data_ptr, x.size, map_dtype(x.dtype))

### SIN / COS ###

def sin(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.sin(
        x._data_ptr, x.size, map_dtype(x.dtype))

def asin(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.asin(
        x._data_ptr, x.size, map_dtype(x.dtype))

def sinh(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.sinh(
        x._data_ptr, x.size, map_dtype(x.dtype))

def asinh(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.asinh(
        x._data_ptr, x.size, map_dtype(x.dtype))

def cos(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.cos(
        x._data_ptr, x.size, map_dtype(x.dtype))

def acos(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.acos(
        x._data_ptr, x.size, map_dtype(x.dtype))

def cosh(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.cosh(
        x._data_ptr, x.size, map_dtype(x.dtype))

def acosh(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.acosh(
        x._data_ptr, x.size, map_dtype(x.dtype))

### TAN / ATAN ###

def tan(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.tan(
        x._data_ptr, x.size, map_dtype(x.dtype))

def tanh(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.tanh(
        x._data_ptr, x.size, map_dtype(x.dtype))

def atan(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.atan(
        x._data_ptr, x.size, map_dtype(x.dtype))

def atanh(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.atanh(
        x._data_ptr, x.size, map_dtype(x.dtype))

def atan2(y: Tensor, x: Tensor, out_shape: tuple[int, ...]) -> int:
    return _nectarml.tensor.elementwise.atan2(
        y._data_ptr, x._data_ptr, 
        y.shape, x.shape, out_shape,
        map_dtype(x.dtype))

### POW ###

def pow(x: Tensor, exponent: float) -> int:
    return _nectarml.tensor.elementwise.pow(
        x._data_ptr, exponent, x.size, map_dtype(x.dtype))

### ABS ###

def abs(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.abs(
        x._data_ptr, x.size, map_dtype(x.dtype))

### ROUNDING ###

def floor(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.floor(
        x._data_ptr, x.size, map_dtype(x.dtype))

def ceil(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.ceil(
        x._data_ptr, x.size, map_dtype(x.dtype))

def round(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.round(
        x._data_ptr, x.size, map_dtype(x.dtype))

### MODULO ###

def fmod(
    x: Tensor, 
    y: Tensor | int | float, 
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(y, int | float):
        return _nectarml.tensor.elementwise.fmod_ts(
            x._data_ptr, y, x.size, map_dtype(x.dtype))
    return _nectarml.tensor.elementwise.fmod(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        map_dtype(x.dtype))

### MIN / MAX ###

def minimum(
    x: Tensor, 
    y: Tensor | int | float, 
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(y, int | float):
        return _nectarml.tensor.elementwise.min_ts(
            x._data_ptr, y, x.size, map_dtype(x.dtype))
    return _nectarml.tensor.elementwise.min(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        map_dtype(x.dtype))
    

def maximum(
    x: Tensor, 
    y: Tensor | int | float,
    out_shape: tuple[int, ...] | None
) -> int:
    if isinstance(y, int | float):
        return _nectarml.tensor.elementwise.max_ts(
            x._data_ptr, y, x.size, map_dtype(x.dtype))
    return _nectarml.tensor.elementwise.max(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        map_dtype(x.dtype))


def clamp(x: Tensor, min_value: float | None, max_value: float | None) -> int:
    min_value = min_value or FLOAT_MIN
    max_value = max_value or FLOAT_MAX
    return _nectarml.tensor.elementwise.clamp(
        x._data_ptr, min_value, max_value, x.size, map_dtype(x.dtype))

### SIGN ###

def sign(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.sign(
        x._data_ptr, x.size, map_dtype(x.dtype))

### COPYSIGN ###

def copysign(x: Tensor, y: Tensor, out_shape: tuple[int, ...]) -> int:
    return _nectarml.tensor.elementwise.copysign(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        map_dtype(x.dtype))

### TRUNCATE ###

def trunc(x: Tensor) -> int:
    return _nectarml.tensor.elementwise.trunc(
        x._data_ptr, x.size, map_dtype(x.dtype))


