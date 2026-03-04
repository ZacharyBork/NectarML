import _nectarml
from nectarml.typing import DTypeLike
from nectarml.cuda.utils import map_dtype

### COMPARISON ###

def equal(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.equal(a_ptr, b_ptr, size, map_dtype(dtype))

def less_than(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.less_than(a_ptr, b_ptr, size, map_dtype(dtype))

def less_than_or_equal(
    a_ptr: int, 
    b_ptr: int, 
    size: int, 
    dtype: DTypeLike
) -> int:
    return _nectarml.less_than_or_equal(a_ptr, b_ptr, size, map_dtype(dtype))

def greater_than(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.greater_than(a_ptr, b_ptr, size, map_dtype(dtype))

def greater_than_or_equal(
    a_ptr: int,
    b_ptr: int, 
    size: int, 
    dtype: DTypeLike
) -> int:
    return _nectarml.greater_than_or_equal(
        a_ptr, b_ptr, size, map_dtype(dtype))

### BASE ###

def add(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.add(a_ptr, b_ptr, size, map_dtype(dtype))

def subtract(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.subtract(a_ptr, b_ptr, size, map_dtype(dtype))

def multiply(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.multiply(a_ptr, b_ptr, size, map_dtype(dtype))

def divide(a_ptr: int, b_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.divide(a_ptr, b_ptr, size, map_dtype(dtype))

### SQRT ###

def sqrt(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.sqrt(x_ptr, size, map_dtype(dtype))

def rsqrt(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.rsqrt(x_ptr, size, map_dtype(dtype))

### EXPONENT ###

def exp(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.exp(x_ptr, size, map_dtype(dtype))

### LOG ###

def log(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.log(x_ptr, size, map_dtype(dtype))

def log2(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.log2(x_ptr, size, map_dtype(dtype))

def log10(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.log10(x_ptr, size, map_dtype(dtype))

### SIN / COS ###

def sin(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.sin(x_ptr, size, map_dtype(dtype))

def asin(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.asin(x_ptr, size, map_dtype(dtype))

def sinh(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.sinh(x_ptr, size, map_dtype(dtype))

def asinh(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.asinh(x_ptr, size, map_dtype(dtype))

def cos(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.cos(x_ptr, size, map_dtype(dtype))

def acos(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.acos(x_ptr, size, map_dtype(dtype))

def cosh(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.cosh(x_ptr, size, map_dtype(dtype))

def acosh(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.acosh(x_ptr, size, map_dtype(dtype))

### TAN / ATAN ###

def tan(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.tan(x_ptr, size, map_dtype(dtype))

def tanh(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.tanh(x_ptr, size, map_dtype(dtype))

def atan(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.atan(x_ptr, size, map_dtype(dtype))

def atanh(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.atanh(x_ptr, size, map_dtype(dtype))

def atan2(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.atan2(x_ptr, size, map_dtype(dtype))

### POW ###

def pow(x_ptr: int, exponent: float, size: int, dtype: DTypeLike) -> int:
    return _nectarml.pow(x_ptr, exponent, size, map_dtype(dtype))

### ABS ###

def abs(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.abs(x_ptr, size, map_dtype(dtype))

### ROUNDING ###

def floor(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.floor(x_ptr, size, map_dtype(dtype))

def ceil(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.ceil(x_ptr, size, map_dtype(dtype))

def round(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.round(x_ptr, size, map_dtype(dtype))

### MODULO ###

def fmod(x_ptr: int, y_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.fmod(x_ptr, y_ptr, size, map_dtype(dtype))

### MIN / MAX ###

def min(x_ptr: int, y_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.min(x_ptr, y_ptr, size, map_dtype(dtype))

def max(x_ptr: int, y_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.max(x_ptr, y_ptr, size, map_dtype(dtype))

### COPYSIGN ###

def copysign(x_ptr: int, y_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.copysign(x_ptr, y_ptr, size, map_dtype(dtype))

### TRUNCATE ###

def trunc(x_ptr: int, size: int, dtype: DTypeLike) -> int:
    return _nectarml.trunc(x_ptr, size, map_dtype(dtype))

