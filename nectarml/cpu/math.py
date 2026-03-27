from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import numpy as np

### COMPARISON ###

def equal(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data == b

def less_than(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data < b

def less_than_or_equal(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data <= b

def greater_than(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data > b

def greater_than_or_equal(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data >= b

### BASIC ###

def add(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data + b

def subtract(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return a.data - b

def multiply(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return  a.data * b

def pow(a: Tensor, exponent: float | int) -> np.ndarray:
    return a.data ** exponent

def matmul(a: Tensor, b: Tensor) -> np.ndarray:
    return np.matmul(a.data, b.data)

def negate(a: Tensor) -> np.ndarray:
    return -a.data

def sign(a: Tensor) -> np.ndarray:
    return np.sign(a.data)

def copysign(a: Tensor, b: Tensor) -> np.ndarray:
    return np.copysign(a.data, b.data)

### OTHER ###

def clamp(
    a: Tensor, 
    min_value: float | None = None, 
    max_value: float | None = None
) -> np.ndarray:
    out = a.data
    if min_value is not None: out = np.maximum(out, min_value)
    if max_value is not None: out = np.minimum(out, max_value)
    return out

def minimum(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return np.minimum(a.data, b)

def maximum(a: Tensor, b: Tensor | int | float) -> np.ndarray:
    if not isinstance(b, int | float): b = b.data
    return np.maximum(a.data, b)
    
def abs(input: Tensor) -> np.ndarray:
    return np.abs(input.data)

def exp(input: Tensor) -> np.ndarray:
    return np.exp(input.data)

def log(input: Tensor) -> np.ndarray:
    return np.log(input.data)

def log2(input: Tensor) -> np.ndarray:
    return np.log2(input.data)

def log10(input: Tensor) -> np.ndarray:
    return np.log10(input.data)

def sqrt(input: Tensor) -> np.ndarray:
    return np.sqrt(input.data)

def rsqrt(input: Tensor) -> np.ndarray:
    return 1 / np.sqrt(input.data)

def sin(input: Tensor) -> np.ndarray:
    return np.sin(input.data)

def asin(input: Tensor) -> np.ndarray:
    return np.asin(input.data)

def sinh(input: Tensor) -> np.ndarray:
    return np.sinh(input.data)

def asinh(input: Tensor) -> np.ndarray:
    return np.asinh(input.data)

def cos(input: Tensor) -> np.ndarray:
    return np.cos(input.data)

def acos(input: Tensor) -> np.ndarray:
    return np.acos(input.data)

def cosh(input: Tensor) -> np.ndarray:
    return np.cosh(input.data)

def acosh(input: Tensor) -> np.ndarray:
    return np.acosh(input.data)

def tan(input: Tensor) -> np.ndarray:
    return np.tan(input.data)

def tanh(input: Tensor) -> np.ndarray:
    return np.tanh(input.data)

def atan(input: Tensor) -> np.ndarray:
    return np.atan(input.data)

def atanh(input: Tensor) -> np.ndarray:
    return np.atanh(input.data)

def atan2(y: Tensor, x: Tensor) -> np.ndarray:
    return np.atan2(y.data, x.data)

