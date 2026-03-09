from collections.abc import Callable

import numpy as np

### BASIC ###

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return  a * b

def pow(a: np.ndarray, exponent: float | int) -> np.ndarray:
    return a ** exponent

def matmul(
    a: np.ndarray, 
    b: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    out = np.matmul(a, b)
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_grad = np.matmul(out_grad, np.swapaxes(b.data, -1, -2))
        b_grad = np.matmul(np.swapaxes(a.data, -1, -2), out_grad)
        return a_grad, b_grad
    return out, _backward

def negate(a: np.ndarray) -> np.ndarray:
    return -a

### OTHER ###

def clamp(
    a: np.ndarray, 
    min_value: float | None = None, 
    max_value: float | None = None
) -> np.ndarray:
    if min_value is not None: a = np.maximum(a, min_value)
    if max_value is not None: a = np.minimum(a, max_value)
    return a

def minimum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

def maximum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)
    
def abs(input: np.ndarray) -> np.ndarray:
    return np.abs(input)

def exp(input: np.ndarray) -> np.ndarray:
    return np.exp(input)

def log(input: np.ndarray) -> np.ndarray:
    return np.log(input)

def log2(input: np.ndarray) -> np.ndarray:
    return np.log2(input)

def log10(input: np.ndarray) -> np.ndarray:
    return np.log10(input)

def sqrt(input: np.ndarray) -> np.ndarray:
    return np.sqrt(input)

def rsqrt(input: np.ndarray) -> np.ndarray:
    return 1 / np.sqrt(input)

def sin(input: np.ndarray) -> np.ndarray:
    return np.sin(input)

def asin(input: np.ndarray) -> np.ndarray:
    return np.asin(input)

def sinh(input: np.ndarray) -> np.ndarray:
    return np.sinh(input)

def asinh(input: np.ndarray) -> np.ndarray:
    return np.asinh(input)

def cos(input: np.ndarray) -> np.ndarray:
    return np.cos(input)

def acos(input: np.ndarray) -> np.ndarray:
    return np.acos(input)

def cosh(input: np.ndarray) -> np.ndarray:
    return np.cosh(input)

def acosh(input: np.ndarray) -> np.ndarray:
    return np.acosh(input)

def tan(input: np.ndarray) -> np.ndarray:
    return np.tan(input)

def tanh(input: np.ndarray) -> np.ndarray:
    return np.tanh(input)

def atan(input: np.ndarray) -> np.ndarray:
    return np.atan(input)

def atanh(input: np.ndarray) -> np.ndarray:
    return np.atanh(input)

def atan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.atan2(y, x)

