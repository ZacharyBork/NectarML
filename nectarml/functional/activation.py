from nectarml import Tensor
from nectarml.functional.common import _eval_core_function
from nectarml._core import activation 

def ReLU(input: Tensor) -> Tensor:
    '''Rectified linear unit activation function.
    
    Returns the input value if the value is >= 0.0, otherwise returns 0.0.
    
    Equation: f(x) = max(0, x)
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.ReLU)

def LeakyReLU(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    '''Leaky rectified linear unit activation function.
    
    Variant of ReLU which allows small non-zero gradients. Helps keep neurons
    active even with very small gradients.
    
    Equation: f(x) = x if x > 0 else negative_slope * x
    
    Args:
        input : The Tensor to apply the activation function to.
        negative_slope : Defines how much variance should be preserved in
            the negative region.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(
        input, activation.LeakyReLU, negative_slope=negative_slope)

def ELU(input: Tensor, alpha: float = 1.0) -> Tensor:
    '''Exponential linear unit activation function.
        
    ReLU activation which applies an exponential curve to negative inputs
    rather than the linear curve of traditional ReLU. Can further help to avoid
    dying gradients, especially in deep networks.
        
    Equation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    
    Args:
        input : The Tensor to apply the activation function to.
        alpha : Multiplier to control saturation level of negative inputs.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.ELU, alpha=alpha)

def SELU(input: Tensor) -> Tensor:
    '''Scaled exponential linear unit activation function.
        
    Similar to ELU activation. SELU automatically normalizes gradients to a
    fixed constant to help stabilize training.
        
    Equation: f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.SELU)

def Sigmoid(input: Tensor) -> Tensor:
    '''Sigmoid activation function.
        
    Maps incoming gradients to an S-shaped curve where (0 <= x <= 1). Useful
    when converting inputs to probabilities for classifier networks.
        
    Equation: f(x) = 1 / (1 + exp(-x))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Sigmoid)

def Tanh(input: Tensor) -> Tensor:
    '''Tanh activation function.
        
    Maps input values along a 0-centered hyperbolic tangent curve.
        
    Equation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Tanh)

def Softmax(input: Tensor, dim: int=-1) -> Tensor:
    '''Softmax activation function.
        
    Generally used in classification tasks. Converts raw output scores into
    [0:1] probabilities.
        
    Equation: f(x) = exp(x_i) / sum(exp(x_j))
    
    Args:
        input : The Tensor to apply the activation function to.
        dim : The dimension along which to compute the activation function.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Softmax, dim=dim)

def LogSoftmax(input: Tensor, dim: int=-1) -> Tensor:
    '''Log softmax activation function.
        
    Computes the logarithm of Softmax(x). Offers improved stability over the
    traditional Softmax activation.
        
    Equation: f(x) = log(exp(x_i) / sum(exp(x_j)))
    
    Args:
        input : The Tensor to apply the activation function to.
        dim : The dimension along which to compute the activation function.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.LogSoftmax, dim=dim)

def GeLU(input: Tensor) -> Tensor:
    '''Gaussian error linear unit activation function.
    
    Smooth approximation of ReLU which scales inputs by the cumulative 
    distribution function of a Gaussian distribution. Enables better gradient
    flow and helps to reduce vanishing gradients in deep networks, and 
    especially in transformer based networks.
    
    Equation: f(x) = x * 0.5 * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
    
    Ref: https://arxiv.org/pdf/1606.08415
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.GeLU)

def SiLU(input: Tensor) -> Tensor:
    '''Sigmoid-weighted linear unit activation function.
        
    Similar to ReLU, SiLU (or Swish) is a smooth, self-gated activation which 
    multiplies inputs by their own Sigmoid transformation. Allows for more
    precise weight accuracy in deep networks and helps to reduce the effects
    of vanishing gradients by allowing small negative weights.
        
    Equation: f(x) = x * (1 / (1 + exp(-x)))
    
    Ref: https://arxiv.org/pdf/1710.05941v1
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.SiLU)

def Swish(input: Tensor) -> Tensor: 
    '''Sigmoid-weighted linear unit activation function.
        
    Similar to ReLU, SiLU (or Swish) is a smooth, self-gated activation which 
    multiplies inputs by their own Sigmoid transformation. Allows for more
    precise weight accuracy in deep networks and helps to reduce the effects
    of vanishing gradients by allowing small negative weights.
        
    Equation: f(x) = x * (1 / (1 + exp(-x)))
    
    Ref: https://arxiv.org/pdf/1710.05941v1
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.SiLU)

def Softplus(input: Tensor) -> Tensor: 
    '''Softplus activation function.
        
    Equation: f(x) = log(1 + exp(x))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Softplus)

def Mish(input: Tensor) -> Tensor:
    '''Softplus activation function.
        
    Equation: f(x) = x * tanh(log(1 + exp(x)))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Mish)

def Hardtanh(
    input: Tensor, 
    min_value: float = -1.0, 
    max_value: float = 1.0
) -> Tensor:
    '''Hardtanh activation function.
        
    Equation: f(x) = max(min_value, min(max_value, x))
    
    Args:
        input : The Tensor to apply the activation function to.
        min_value : The minimum allowable value. Values below this will be
            clamped to this value.
        max_value : The maximum allowable value. Values above this will be
            clamped to this value.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(
        input, activation.Hardtanh, min_value=min_value, max_value=max_value)

def Hardsigmoid(input: Tensor) -> Tensor:
    '''Hardsigmoid activation function.
        
    Equation: f(x) = max(0, min(1, (x + 1) / 2))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Hardsigmoid)

def Hardswish(input: Tensor) -> Tensor:
    '''Hardswish activation function.
        
    Equation: f(x) = x * max(0, min(1, (x + 1) / 2))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Hardswish)

def Softsign(input: Tensor) -> Tensor:
    '''Softsign activation function.
        
    Equation: f(x) = x / (1 + |x|)
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Softsign)

def Softmin(input: Tensor) -> Tensor:
    '''Softmin activation function.
        
    Equation: f(x) = exp(-x_i) / sum(exp(-x_j))
    
    Args:
        input : The Tensor to apply the activation function to.
        
    Returns:
        Tensor : The resulting Tensor from the activation function.
    '''
    return _eval_core_function(input, activation.Softmin)

