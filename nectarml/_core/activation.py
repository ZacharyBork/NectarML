from collections.abc import Callable

import numpy as np

def ReLU(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Rectified linear unit activation function.
    
    Returns the input value if the value is >= 0.0, otherwise returns 0.0.
    
    Equation: f(x) = max(0, x)
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    zeros = np.zeros_like(input)
    out = np.maximum(zeros, input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (input > zeros).astype(input.dtype) * out_grad
    return out, _backward

def LeakyReLU(
    input: np.ndarray, 
    negative_slope: float = 0.01
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Leaky rectified linear unit activation function.
    
    Variant of ReLU which allows small non-zero gradients. Helps keep neurons
    active even with very small gradients.
    
    Equation: f(x) = x if x > 0 else negative_slope * x
    
    Args:
        input : The np.ndarray to apply the activation function to.
        negative_slope : Defines how much variance should be preserved in
            the negative region.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    out = np.where(input > 0, input, negative_slope * input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1.0, negative_slope) * out_grad
    return out, _backward

def ELU(
    input: np.ndarray, 
    alpha: float = 1.0
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Exponential linear unit activation function.
        
    ReLU activation which applies an exponential curve to negative inputs
    rather than the linear curve of traditional ReLU. Can further help to avoid
    dying gradients, especially in deep networks.
        
    Equation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    
    Args:
        input : The np.ndarray to apply the activation function to.
        alpha : Multiplier to control saturation level of negative inputs.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    out = np.where(input > 0, input, alpha * (np.exp(input) - 1.0))
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = np.where(input > 0, 1.0, alpha * np.exp(input))
        return grad * out_grad
    return out, _backward

def SELU(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Scaled exponential linear unit activation function.
        
    Similar to ELU activation. SELU automatically normalizes gradients to a
    fixed constant to help stabilize training.
        
    Equation: f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    alpha = 1.0507
    scale = 1.050
    out = scale * np.where(input > 0, input, alpha * (np.exp(input) - 1))
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = np.where(input > 0, 1.0, alpha * np.exp(input))
        return (1 / alpha) * grad * out_grad
    return out, _backward

def Sigmoid(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Sigmoid activation function.
        
    Maps incoming gradients to an S-shaped curve where (0 <= x <= 1). Useful
    when converting inputs to probabilities for classifier networks.
        
    Equation: f(x) = 1 / (1 + exp(-x))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    out = 1 / (1 + np.exp(-input))
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return out * (1 - out) * out_grad
    return out, _backward
    
def Tanh(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Tanh activation function.
        
    Maps input values along a 0-centered hyperbolic tangent curve.
        
    Equation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    exp = np.exp(input)
    inv_exp = np.exp(-input)
    out = (exp - inv_exp) / (exp + inv_exp)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (1 - out ** 2) * out_grad 
    return out, _backward

def Softmax(
    input: np.ndarray, 
    dim: int=-1
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Softmax activation function.
        
    Generally used in classification tasks. Converts raw output scores into
    [0:1] probabilities.
        
    Equation: f(x) = exp(x_i) / sum(exp(x_j))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        dim : The dimension along which to compute the activation function.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    exp_x = np.exp(input - input.max(dim=dim, keepdims=True))
    out = exp_x / np.sum(exp_x, dim=dim, keepdims=True)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        S = np.sum(out_grad * out, dim=dim, keepdims=True)
        return out * (out_grad - S)
    return out, _backward

def LogSoftmax(
    input: np.ndarray, 
    dim: int=-1
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Log softmax activation function.
        
    Computes the logarithm of Softmax(x). Offers improved stability over the
    traditional Softmax activation.
        
    Equation: f(x) = log(exp(x_i) / sum(exp(x_j)))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        dim : The dimension along which to compute the activation function.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    exp_x = np.exp(input - input.max(dim=dim, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, dim=dim, keepdims=True)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = out_grad - softmax_x * np.sum(out_grad, axis=dim, keepdims=True)
        return grad
    return np.log(softmax_x), _backward

def GeLU(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Gaussian error linear unit activation function.
    
    Smooth approximation of ReLU which scales inputs by the cumulative 
    distribution function of a Gaussian distribution. Enables better gradient
    flow and helps to reduce vanishing gradients in deep networks, and 
    especially in transformer based networks.
    
    Equation: f(x) = x * 0.5 * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
    
    Ref: https://arxiv.org/pdf/1606.08415
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    inner = 0.7978845608 * (input + 0.044715 * input ** 3)
    exp = np.exp(inner)
    inv_exp = np.exp(-inner)
    tanh_inner = (exp - inv_exp) / (exp + inv_exp)
    out = input * 0.5 * (1 + tanh_inner)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        d_inner = 0.7978845608 * (1.0 + 0.044715 * 3 * input ** 2)
        grad = 0.5 * ((1+tanh_inner) + input * (1-tanh_inner ** 2) * d_inner)
        return grad * out_grad        
    return out, _backward

def SiLU(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Sigmoid-weighted linear unit activation function.
        
    Similar to ReLU, SiLU (or Swish) is a smooth, self-gated activation which 
    multiplies inputs by their own Sigmoid transformation. Allows for more
    precise weight accuracy in deep networks and helps to reduce the effects
    of vanishing gradients by allowing small negative weights.
        
    Equation: f(x) = x * (1 / (1 + exp(-x)))
    
    Ref: https://arxiv.org/pdf/1710.05941v1
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    sigmoid = 1 / (1 + np.exp(-input))
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return (sigmoid + input * sigmoid * (1 - sigmoid)) * out_grad
    return input * sigmoid, _backward

def Softplus(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Softplus activation function.
        
    Equation: f(x) = log(1 + exp(x))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    exp_in = np.exp(input)
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        return exp_in / (1 + exp_in) * out_grad
    return np.log(1 + exp_in), _backward

def Mish(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Softplus activation function.
        
    Equation: f(x) = x * tanh(log(1 + exp(x)))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    exp_in = np.exp(input)
    softplus = np.log(1 + exp_in)

    sp_exp = np.exp(softplus)
    inv_sp_exp = np.exp(-softplus)
    tanh = (sp_exp - inv_sp_exp) / (sp_exp + inv_sp_exp)
    
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        d_softplus = exp_in / (1 + exp_in)
        d_tanh = (1 - tanh ** 2)
        grad = tanh + input * d_tanh * d_softplus
        return grad * out_grad

    return input * tanh, _backward

def Hardtanh(
    input: np.ndarray, 
    min_value: float = -1.0, 
    max_value: float = 1.0
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Hardtanh activation function.
        
    Equation: f(x) = max(min_value, min(max_value, x))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        min_value : The minimum allowable value. Values below this will be
            clamped to this value.
        max_value : The maximum allowable value. Values above this will be
            clamped to this value.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    _min = np.full(
        (), fill_value=min_value, dtype=input.dtype, device=input.device)
    _max = np.full(
        (), fill_value=max_value, dtype=input.dtype, device=input.device)
    out = np.maximum(_min, np.minimum(_max, input))
    
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        mask = ((_min <= input) & (input <= _max)).astype(input.dtype)
        return mask * out_grad
    
    return out, _backward

def Hardsigmoid(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Hardsigmoid activation function.
        
    Equation: f(x) = max(0, min(1, (x + 1) / 2))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    _zeros = np.zeros((), dtype=input.dtype, device=input.device)
    _ones = np.ones((), dtype=input.dtype, device=input.device)
    out = np.maximum(_zeros, np.minimum(_ones, (input + 1) / 2))
    
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        mask = ((_zeros < out) & (out < _ones)).astype(input.dtype)
        return 0.5 * mask * out_grad
    
    return out, _backward

def Hardswish(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Hardswish activation function.
        
    Equation: f(x) = x * max(0, min(1, (x + 1) / 2))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    _zeros = np.zeros((), dtype=input.dtype, device=input.device)
    _ones = np.ones((), dtype=input.dtype, device=input.device)
    sigmoid = np.maximum(_zeros, np.minimum(_ones, (input + 1) / 2))
    
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        mask = ((_zeros < sigmoid) & (sigmoid < _ones)).astype(input.dtype)
        return (sigmoid + input * 0.5 * mask) * out_grad
    
    return input * sigmoid, _backward

def Softsign(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Softsign activation function.
        
    Equation: f(x) = x / (1 + |x|)
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    out = input / (1 + np.abs(input))
    def _backward(out_grad: np.ndarray) -> np.ndarray:
        grad = 1 / (1 + np.abs(input)) ** 2 
        return grad * out_grad
    return out, _backward

def Softmin(
    input: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    '''Softmin activation function.
        
    Equation: f(x) = exp(-x_i) / sum(exp(-x_j))
    
    Args:
        input : The np.ndarray to apply the activation function to.
        
    Returns:
        tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] : The resulting 
            np.ndarray from the activation function, and the corresponding
            backpropagation function with signature:

                def _backward(out_grad: np.ndarray) -> np.ndarray:
                    return activation_grad * out_grad
    '''
    out, _backward = Softmax(-input)
    def _backward_wrapper(out_grad: np.ndarray) -> np.ndarray:
        return -_backward(out_grad)
    return out, _backward_wrapper

