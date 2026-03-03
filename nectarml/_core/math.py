from collections.abc import Callable

import numpy as np

### BASIC ###

def add(
    a: np.ndarray, 
    b: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = a + b
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad
    return out, _backward

def subtract(
    a: np.ndarray, 
    b: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = a - b
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad
    return out, _backward

def multiply(
    a: np.ndarray, 
    b: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    out = a * b
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_grad = b * out_grad
        b_grad = a * out_grad
        return a_grad, b_grad
    return out, _backward

def pow(
    a: np.ndarray, 
    exponent: float | int
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    out = a ** exponent
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return exponent * (a**(exponent-1)) * out_grad
    return out, _backward

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

def negate(
    a: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = -a
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return -out_grad
    return out, _backward

### OTHER ###

def minimum(
    a: np.ndarray, 
    b: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    out = np.minimum(a, b)
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_grad = (a <= b).astype(a.dtype) * out_grad
        b_grad = (b < a).astype(b.dtype) * out_grad
        return a_grad, b_grad
    return out, _backward

def maximum(
    a: np.ndarray, 
    b: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    out = np.maximum(a, b)
    def _backward(out_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_grad = (a >= b).astype(a.dtype) * out_grad
        b_grad = (b > a).astype(b.dtype) * out_grad
        return a_grad, b_grad
    return out, _backward
    
def abs(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.abs(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.sign(input) * out_grad
    return out, _backward

def exp(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.exp(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out * out_grad
    return out, _backward

def log(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.log(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (1 / input) * out_grad
    return out, _backward

def log2(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.log2(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / (input * np.log(2))
    return out, _backward

def log2(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.log2(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / (input * np.log(10))
    return out, _backward

def sqrt(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.sqrt(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (1 / (2 * out)) * out_grad
    return out, _backward

def sin(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.sin(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.cos(input) * out_grad
    return out, _backward

def asin(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.asin(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / np.sqrt(np.maximum(1 - input**2, 1e-7))
    return out, _backward

def sinh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.sinh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.cosh(input) * out_grad
    return out, _backward

def asinh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.asinh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / np.sqrt(input**2 + 1)
    return out, _backward

def cos(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.cos(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return -np.sin(input) * out_grad
    return out, _backward

def acos(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.acos(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return -out_grad / np.sqrt(np.maximum(1 - input**2, 1e-7))
    return out, _backward

def cosh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.cosh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.sinh(input) * out_grad
    return out, _backward

def acosh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.acosh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / np.sqrt(np.maximum(input**2 - 1, 1e-7))
    return out, _backward

def tan(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.tan(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad * (1 + out**2)
    return out, _backward

def tanh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.tanh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (1 - out ** 2) * out_grad
    return out, _backward

def atan(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.atan(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / (1 + input**2)
    return out, _backward

def atanh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.atanh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out_grad / np.maximum(1 - input**2, 1e-7)
    return out, _backward

def atan2(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.atan2(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        denom = input**2 + out**2
        grad_x = out_grad * -out / np.maximum(denom, 1e-7)
        grad_y = out_grad * input / np.maximum(denom, 1e-7)
        return grad_y, grad_x
    return out, _backward

