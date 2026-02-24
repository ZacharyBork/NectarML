from nectarml import Tensor
from nectarml.functional.common import _eval_core_function
from nectarml._core import math
  
def abs(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.abs)

def exp(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.exp)

def log(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.log)

def sqrt(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.sqrt)

def sin(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.sin)

def cos(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.cos)

def cosh(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.cosh)

def tanh(input: Tensor) -> Tensor:
    return _eval_core_function(input, math.tanh)

def sigmoid(input: Tensor) -> Tensor:
    return (exp(-input) + 1) ** -1
