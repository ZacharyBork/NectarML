from collections.abc import Callable

import numpy as np

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

def cos(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.cos(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return -np.sin(input) * out_grad
    return out, _backward

def cosh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.cosh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.sinh(input) * out_grad
    return out, _backward

def tanh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    out = np.tanh(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (1 - out ** 2) * out_grad
    return out, _backward

