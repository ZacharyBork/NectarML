from nectarml.tensor import Tensor
  
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

def clamp(
    a: Tensor, 
    min_value: float | None = None, 
    max_value: float | None = None
) -> Tensor:
    return a.clamp(min_value=min_value, max_value=max_value)
  
def minimum(a: Tensor, b: Tensor) -> Tensor: 
    return a.minimum(b)

def maximum(a: Tensor, b: Tensor) -> Tensor: 
    return a.maximum(b)
  
def abs(input: Tensor) -> Tensor: 
    return input.abs()

def exp(input: Tensor) -> Tensor: 
    return input.exp()

def log(input: Tensor) -> Tensor: 
    return input.log()

def log2(input: Tensor) -> Tensor: 
    return input.log2()

def log10(input: Tensor) -> Tensor: 
    return input.log10()

def sqrt(input: Tensor) -> Tensor: 
    return input.sqrt()

def rsqrt(input: Tensor) -> Tensor: 
    return input.rsqrt()

def sin(input: Tensor) -> Tensor:
    return input.sin()

def asin(input: Tensor) -> Tensor:
    return input.asin()

def sinh(input: Tensor) -> Tensor:
    return input.sinh()

def asinh(input: Tensor) -> Tensor:
    return input.asinh()

def cos(input: Tensor) -> Tensor:
    return input.cos()

def acos(input: Tensor) -> Tensor:
    return input.acos()

def cosh(input: Tensor) -> Tensor:
    return input.cosh()

def acosh(input: Tensor) -> Tensor:
    return input.acosh()

def tan(input: Tensor) -> Tensor:
    return input.tan()

def tanh(input: Tensor) -> Tensor:
    return input.tanh()

def atan(input: Tensor) -> Tensor:
    return input.atan()

def atanh(input: Tensor) -> Tensor:
    return input.atanh()

def atan2(y: Tensor, x: Tensor) -> Tensor:
    return x.atan2(y)

def sigmoid(input: Tensor) -> Tensor:
    return (exp(-input) + 1) ** -1
