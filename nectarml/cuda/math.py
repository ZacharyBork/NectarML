from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype
from nectarml.constants import FLOAT_MIN, FLOAT_MAX

### COMPARISON ###

def equal(a: Tensor, b: Tensor) -> int:
    return _nectarml.equal(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def less_than(a: Tensor, b: Tensor) -> int:
    return _nectarml.less_than(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def less_than_or_equal(a: Tensor, b: Tensor) -> int:
    return _nectarml.less_than_or_equal(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def greater_than(a: Tensor, b: Tensor) -> int:
    return _nectarml.greater_than(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def greater_than_or_equal(a: Tensor, b: Tensor) -> int:
    return _nectarml.greater_than_or_equal(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

### BASE ###

def add(a: Tensor, b: Tensor) -> int:
    return _nectarml.add(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def subtract(a: Tensor, b: Tensor) -> int:
    return _nectarml.subtract(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def multiply(a: Tensor, b: Tensor) -> int:
    return _nectarml.multiply(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def divide(a: Tensor, b: Tensor) -> int:
    return _nectarml.divide(
        a._data_ptr, b._data_ptr, a.size, map_dtype(a.dtype))

def negate(x: Tensor) -> int:
    return _nectarml.negate(x._data_ptr, x.size, map_dtype(x.dtype))

### SQRT ###

def sqrt(x: Tensor) -> int:
    return _nectarml.sqrt(x._data_ptr, x.size, map_dtype(x.dtype))

def rsqrt(x: Tensor) -> int:
    return _nectarml.rsqrt(x._data_ptr, x.size, map_dtype(x.dtype))

### EXPONENT ###

def exp(x: Tensor) -> int:
    return _nectarml.exp(x._data_ptr, x.size, map_dtype(x.dtype))

### LOG ###

def log(x: Tensor) -> int:
    return _nectarml.log(x._data_ptr, x.size, map_dtype(x.dtype))

def log2(x: Tensor) -> int:
    return _nectarml.log2(x._data_ptr, x.size, map_dtype(x.dtype))

def log10(x: Tensor) -> int:
    return _nectarml.log10(x._data_ptr, x.size, map_dtype(x.dtype))

### SIN / COS ###

def sin(x: Tensor) -> int:
    return _nectarml.sin(x._data_ptr, x.size, map_dtype(x.dtype))

def asin(x: Tensor) -> int:
    return _nectarml.asin(x._data_ptr, x.size, map_dtype(x.dtype))

def sinh(x: Tensor) -> int:
    return _nectarml.sinh(x._data_ptr, x.size, map_dtype(x.dtype))

def asinh(x: Tensor) -> int:
    return _nectarml.asinh(x._data_ptr, x.size, map_dtype(x.dtype))

def cos(x: Tensor) -> int:
    return _nectarml.cos(x._data_ptr, x.size, map_dtype(x.dtype))

def acos(x: Tensor) -> int:
    return _nectarml.acos(x._data_ptr, x.size, map_dtype(x.dtype))

def cosh(x: Tensor) -> int:
    return _nectarml.cosh(x._data_ptr, x.size, map_dtype(x.dtype))

def acosh(x: Tensor) -> int:
    return _nectarml.acosh(x._data_ptr, x.size, map_dtype(x.dtype))

### TAN / ATAN ###

def tan(x: Tensor) -> int:
    return _nectarml.tan(x._data_ptr, x.size, map_dtype(x.dtype))

def tanh(x: Tensor) -> int:
    return _nectarml.tanh(x._data_ptr, x.size, map_dtype(x.dtype))

def atan(x: Tensor) -> int:
    return _nectarml.atan(x._data_ptr, x.size, map_dtype(x.dtype))

def atanh(x: Tensor) -> int:
    return _nectarml.atanh(x._data_ptr, x.size, map_dtype(x.dtype))

def atan2(y: Tensor, x: Tensor) -> int:
    return _nectarml.atan2(
        y._data_ptr, x._data_ptr, x.size, map_dtype(x.dtype))

### POW ###

def pow(x: Tensor, exponent: float) -> int:
    return _nectarml.pow(x._data_ptr, exponent, x.size, map_dtype(x.dtype))

### ABS ###

def abs(x: Tensor) -> int:
    return _nectarml.abs(x._data_ptr, x.size, map_dtype(x.dtype))

### ROUNDING ###

def floor(x: Tensor) -> int:
    return _nectarml.floor(x._data_ptr, x.size, map_dtype(x.dtype))

def ceil(x: Tensor) -> int:
    return _nectarml.ceil(x._data_ptr, x.size, map_dtype(x.dtype))

def round(x: Tensor) -> int:
    return _nectarml.round(x._data_ptr, x.size, map_dtype(x.dtype))

### MODULO ###

def fmod(x: Tensor, y: Tensor) -> int:
    return _nectarml.fmod(x._data_ptr, y._data_ptr, x.size, map_dtype(x.dtype))

### MIN / MAX ###

def minimum(x: Tensor, y: Tensor) -> int:
    return _nectarml.min(x._data_ptr, y._data_ptr, x.size, map_dtype(x.dtype))

def maximum(x: Tensor, y: Tensor) -> int:
    return _nectarml.max(x._data_ptr, y._data_ptr, x.size, map_dtype(x.dtype))

def clamp(x: Tensor, min_value: float | None, max_value: float | None) -> int:
    min_value = min_value or FLOAT_MIN
    max_value = max_value or FLOAT_MAX
    return _nectarml.clamp(
        x._data_ptr, min_value, max_value, x.size, map_dtype(x.dtype))

### COPYSIGN ###

def copysign(x: Tensor, y: Tensor) -> int:
    return _nectarml.copysign(
        x._data_ptr, y._data_ptr, x.size, map_dtype(x.dtype))

### TRUNCATE ###

def trunc(x: Tensor) -> int:
    return _nectarml.trunc(x._data_ptr, x.size, map_dtype(x.dtype))


