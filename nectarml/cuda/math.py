from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import builtins

import _nectarml
from nectarml.constants import FLOAT_MIN, FLOAT_MAX

### COMPARISON ###

def equal(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.equal_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)

def not_equal(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.not_equal_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.not_equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)

def less_than(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.less_than_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.less_than(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)

def less_than_or_equal(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.less_than_or_equal_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.less_than_or_equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)

def greater_than(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.greater_than_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.greater_than(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)

def greater_than_or_equal(
    a: Tensor,
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.greater_than_or_equal_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.greater_than_or_equal(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)

### BASE ###

def add(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float, 
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.add_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.add(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)
    

def subtract(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float, 
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.subtract_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.subtract(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)
    

def multiply(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.multiply_ts(
         a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.multiply(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)
    
def divide(
    a: Tensor, 
    b: Tensor | builtins.int | builtins.float, 
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(b, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.divide_ts(
            a._data_ptr, b, a.size, a.dtype.cuda)
    return _nectarml.tensor.elementwise.divide(
        a._data_ptr, b._data_ptr, 
        a.shape, b.shape, out_shape,
        a.dtype.cuda)
    

def negate(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.negate(
        x._data_ptr, x.size, x.dtype.cuda)

### SQRT ###

def sqrt(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.sqrt(
        x._data_ptr, x.size, x.dtype.cuda)

def rsqrt(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.rsqrt(
        x._data_ptr, x.size, x.dtype.cuda)

### EXPONENT ###

def exp(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.exp(
        x._data_ptr, x.size, x.dtype.cuda)

### LOG ###

def log(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.log(
        x._data_ptr, x.size, x.dtype.cuda)

def log2(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.log2(
        x._data_ptr, x.size, x.dtype.cuda)

def log10(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.log10(
        x._data_ptr, x.size, x.dtype.cuda)

### SIN / COS ###

def sin(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.sin(
        x._data_ptr, x.size, x.dtype.cuda)

def asin(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.asin(
        x._data_ptr, x.size, x.dtype.cuda)

def sinh(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.sinh(
        x._data_ptr, x.size, x.dtype.cuda)

def asinh(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.asinh(
        x._data_ptr, x.size, x.dtype.cuda)

def cos(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.cos(
        x._data_ptr, x.size, x.dtype.cuda)

def acos(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.acos(
        x._data_ptr, x.size, x.dtype.cuda)

def cosh(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.cosh(
        x._data_ptr, x.size, x.dtype.cuda)

def acosh(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.acosh(
        x._data_ptr, x.size, x.dtype.cuda)

### TAN / ATAN ###

def tan(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.tan(
        x._data_ptr, x.size, x.dtype.cuda)

def tanh(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.tanh(
        x._data_ptr, x.size, x.dtype.cuda)

def atan(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.atan(
        x._data_ptr, x.size, x.dtype.cuda)

def atanh(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.atanh(
        x._data_ptr, x.size, x.dtype.cuda)

def atan2(
    y: Tensor, 
    x: Tensor, 
    out_shape: tuple[builtins.int, ...]
) -> builtins.int:
    return _nectarml.tensor.elementwise.atan2(
        y._data_ptr, x._data_ptr, 
        y.shape, x.shape, out_shape,
        x.dtype.cuda)

### POW ###

def pow(x: Tensor, exponent: builtins.float) -> builtins.int:
    return _nectarml.tensor.elementwise.pow(
        x._data_ptr, exponent, x.size, x.dtype.cuda)

### ABS ###

def abs(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.abs(
        x._data_ptr, x.size, x.dtype.cuda)

### ROUNDING ###

def floor(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.floor(
        x._data_ptr, x.size, x.dtype.cuda)

def ceil(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.ceil(
        x._data_ptr, x.size, x.dtype.cuda)

def round(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.round(
        x._data_ptr, x.size, x.dtype.cuda)

### MODULO ###

def fmod(
    x: Tensor, 
    y: Tensor | builtins.int | builtins.float, 
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(y, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.fmod_ts(
            x._data_ptr, y, x.size, x.dtype.cuda)
    return _nectarml.tensor.elementwise.fmod(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        x.dtype.cuda)

### MIN / MAX ###

def minimum(
    x: Tensor, 
    y: Tensor | builtins.int | builtins.float, 
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(y, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.min_ts(
            x._data_ptr, y, x.size, x.dtype.cuda)
    return _nectarml.tensor.elementwise.min(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        x.dtype.cuda)
    

def maximum(
    x: Tensor, 
    y: Tensor | builtins.int | builtins.float,
    out_shape: tuple[builtins.int, ...] | None
) -> builtins.int:
    if isinstance(y, builtins.int | builtins.float):
        return _nectarml.tensor.elementwise.max_ts(
            x._data_ptr, y, x.size, x.dtype.cuda)
    return _nectarml.tensor.elementwise.max(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        x.dtype.cuda)


def clamp(
    x: Tensor, 
    min_value: builtins.float | None, 
    max_value: builtins.float | None
) -> builtins.int:
    min_value = min_value or FLOAT_MIN
    max_value = max_value or FLOAT_MAX
    return _nectarml.tensor.elementwise.clamp(
        x._data_ptr, min_value, max_value, x.size, x.dtype.cuda)

### SIGN ###

def sign(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.sign(
        x._data_ptr, x.size, x.dtype.cuda)

### COPYSIGN ###

def copysign(
    x: Tensor, 
    y: Tensor, 
    out_shape: tuple[builtins.int, ...]
) -> builtins.int:
    return _nectarml.tensor.elementwise.copysign(
        x._data_ptr, y._data_ptr, 
        x.shape, y.shape, out_shape,
        x.dtype.cuda)

### TRUNCATE ###

def trunc(x: Tensor) -> builtins.int:
    return _nectarml.tensor.elementwise.trunc(
        x._data_ptr, x.size, x.dtype.cuda)


