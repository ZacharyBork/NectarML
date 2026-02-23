import numpy as np

from nectarml import Tensor
from nectarml.functional.common import _wrapper_base

def abs(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.abs, np.sign)
    
def exp(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.exp, np.exp)

def log(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.log, lambda x: 1 / x)

def sqrt(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))

def sin(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.sin, np.cos)

def cos(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.cos, lambda x: -np.sin(x))

def cosh(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.cosh, np.sinh)

def tanh(input: Tensor) -> Tensor: 
    return _wrapper_base(input, np.tanh, lambda x: 1 - np.tanh(x) ** 2)

def sigmoid(input: Tensor) -> Tensor:
    return (exp(-input) + 1) ** -1
