from nectarml.core import Tensor
  
### BASIC ###
  
def add(a: Tensor, b: Tensor | int | float) -> Tensor:
    '''Adds a value to a tensor and returns the result.
    
    Args:
        a : The tensor to add to.
        b : The value to add. Can be another tensor, an integer, or a floating
            point number.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a + b

def subtract(a: Tensor, b: Tensor | int | float) -> Tensor:
    '''Subtracts a value to a tensor and returns the result.
    
    Args:
        a : The tensor to subtract from.
        b : The value to subtract. Can be another tensor, an integer, or a 
            floating point number.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a - b

def multiply(a: Tensor, b: Tensor | int | float) -> Tensor:
    '''Multiplies a tensor by a value and returns the result.
    
    Args:
        a : The tensor to multiply.
        b : The value to multiply it with. Can be another tensor, an integer, 
            or a floating point number.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a * b

def pow(a: Tensor, exponent: int | float) -> Tensor:
    '''Raises a tensors data to the provided exponent and returns the result.
    
    Args:
        a        : The tensor to raise.
        exponent : The exponent to raise the tensor's values to.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a**exponent

def matmul(a: Tensor, b: Tensor) -> Tensor:
    '''Performs matrix multiplication between two tensors, returns the result.
    
    Args:
        a : The multiplicand tensor.
        b : The multiplier tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a @ b

def negate(a: Tensor) -> Tensor:
    '''Negates a given tensors data and returns the result.
    
    Args:
        a : The tensor to negate.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return -a
  
### ROUNDING ###
  
def floor(input: Tensor) -> Tensor:
    '''Takes the floor of a given tensors data and returns the result.
    
    This does not affect the DType of the tensor. The returned tensor's DType
    will be the same as the input tensor's DType.
    
    Args:
        input : The input tensor.

    Returns:
        tensor : The resulting Tensor.
    '''
    return input.floor()
  
def ceil(input: Tensor) -> Tensor:
    '''Takes the ceiling of a given tensors data and returns the result.
    
    This does not affect the DType of the tensor. The returned tensor's DType
    will be the same as the input tensor's DType.
    
    Args:
        input : The input tensor.

    Returns:
        tensor : The resulting Tensor.
    '''
    return input.ceil()

def round(input: Tensor, precision: int = 0) -> Tensor:
    '''Rounds a tensor's values to the nearest whole number.
    
    This does not affect the DType of the tensor. The returned tensor's DType
    will be the same as the input tensor's DType.
    
    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.round(precision)
  
### OTHER ###

def clamp(
    a: Tensor, 
    min_value: float | None = None, 
    max_value: float | None = None
) -> Tensor:
    '''Clamps a tensor's values beween min/max values and returns the result.

    Args:
        a         : The tensor to clamp.
        min_value : The minimum value to allow, or None to use the input
                    tensor DType's minimum possible value.
        max_value : The maximum value to allow, or None to use the input
                    tensor DType's maximum possible value.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a.clamp(min_value=min_value, max_value=max_value)
  
def minimum(a: Tensor, b: Tensor | int | float) -> Tensor:
    '''Returns the minumum of a tensor's values and a provided value.

    Args:
        a : The tensor to compare.
        b : The value to compare it against. Can be another tensor, an integer,
            or a floating point number.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a.minimum(b)

def maximum(a: Tensor, b: Tensor | int | float) -> Tensor: 
    '''Returns the maximum of a tensor's values and a provided value.

    Args:
        a : The tensor to compare.
        b : The value to compare it against. Can be another tensor, an integer,
            or a floating point number.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return a.maximum(b)
  
def abs(input: Tensor) -> Tensor:
    '''Returns the absolute of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.abs()

def exp(input: Tensor) -> Tensor: 
    '''Returns the exponent of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.exp()

def log(input: Tensor) -> Tensor: 
    '''Returns the log of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.log()

def log2(input: Tensor) -> Tensor: 
    '''Returns the log2 of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.log2()

def log10(input: Tensor) -> Tensor: 
    '''Returns the log10 of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.log10()

def sqrt(input: Tensor) -> Tensor: 
    '''Returns the square root of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.sqrt()

def rsqrt(input: Tensor) -> Tensor: 
    '''Returns the reciprocal square root of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.rsqrt()

def sin(input: Tensor) -> Tensor:
    '''Returns the sine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.sin()

def asin(input: Tensor) -> Tensor:
    '''Returns the arc sine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.asin()

def sinh(input: Tensor) -> Tensor:
    '''Returns the hyperbolic sine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.sinh()

def asinh(input: Tensor) -> Tensor:
    '''Returns the hyperbolic arc sine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.asinh()

def cos(input: Tensor) -> Tensor:
    '''Returns the cosine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.cos()

def acos(input: Tensor) -> Tensor:
    '''Returns the arc cosine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.acos()

def cosh(input: Tensor) -> Tensor:
    '''Returns the hyperbolic cosine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.cosh()

def acosh(input: Tensor) -> Tensor:
    '''Returns the hyperbolic arc cosine of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.acosh()

def tan(input: Tensor) -> Tensor:
    '''Returns the tangent of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.tan()

def tanh(input: Tensor) -> Tensor:
    '''Returns the hyperbolic tangent of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.tanh()

def atan(input: Tensor) -> Tensor:
    '''Returns the arc tangent of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.atan()

def atanh(input: Tensor) -> Tensor:
    '''Returns the hyperbolic arc tangent of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return input.atanh()

def atan2(y: Tensor, x: Tensor) -> Tensor:
    '''Returns the arc tangent^2 of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return x.atan2(y)

def sigmoid(input: Tensor) -> Tensor:
    '''Returns the sigmoid of a given tensor's values.

    Args:
        input : The input tensor.
        
    Returns:
        tensor : The resulting Tensor.
    '''
    return (exp(-input) + 1) ** -1
