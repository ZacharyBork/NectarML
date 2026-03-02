from nectarml import Tensor
from nectarml.functional.common import _eval_core_function
from nectarml._core import math
  
### BASIC ###
  
def add(a: Tensor, b: Tensor | int | float) -> Tensor:
    return a + b

def subtract(a: Tensor, b: Tensor | int | float) -> Tensor:
    return a - b

def multiply(a: Tensor, b: Tensor | int | float) -> Tensor:
    return a * b

def pow(a: Tensor, exponent: int | float) -> Tensor:
    return a**exponent

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b

def negate(a: Tensor) -> Tensor:
    return -a
  
### OTHER ###
  
def minimum(a: Tensor, b: Tensor) -> Tensor:
    out_data, _backward = math.minimum(a.data, b.data)
    out = a._build_output_tensor(out_data, (a, b))
    def _backward_hook():
        a_grad, b_grad = _backward(out.grad)
        if a.requires_grad: a.grad += a_grad
        if b.requires_grad: b.grad += b_grad
    out._backward = _backward_hook
    return out

def maximum(a: Tensor, b: Tensor) -> Tensor:
    out_data, _backward = math.maximum(a.data, b.data)
    out = a._build_output_tensor(out_data, (a, b))
    def _backward_hook():
        a_grad, b_grad = _backward(out.grad)
        if a.requires_grad: a.grad += a_grad
        if b.requires_grad: b.grad += b_grad
    out._backward = _backward_hook
    return out
  
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
