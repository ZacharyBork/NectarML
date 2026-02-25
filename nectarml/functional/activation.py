from nectarml import Tensor, zeros_like, full, zeros, ones
import nectarml.functional as F

def ReLU(input: Tensor) -> Tensor:
    return F.maximum(zeros_like(input), input)

def LeakyReLU(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    return F.where(input > 0, input, negative_slope * input)

def ELU(input: Tensor, alpha: float = 1.0) -> Tensor:
    return F.where(input > 0, input, alpha * (F.exp(input) - 1.0))

def SELU(input: Tensor) -> Tensor:
    return 1.0507 * ELU(input, alpha=1.6733)

def Sigmoid(input: Tensor) -> Tensor:
    return 1 / (1 + F.exp(-input))

def Tanh(input: Tensor) -> Tensor:
    exp = F.exp(input)
    inv_exp = F.exp(-input)
    return (exp - inv_exp) / (exp + inv_exp)

def Softmax(input: Tensor, dim: int=-1) -> Tensor:
    exp_x = F.exp(input - input.max(dim=dim, keepdims=True))
    return exp_x / exp_x.sum(dim=dim, keepdims=True)

def LogSoftmax(input: Tensor, dim: int=-1) -> Tensor:
    return F.log(Softmax(input, dim=dim))

def GeLU(input: Tensor) -> Tensor:
    inner = 0.7978845608 * (input + 0.044715 * input ** 3)
    return input * 0.5 * (1 + Tanh(inner))

def SiLU(input: Tensor) -> Tensor:
    return input * Sigmoid(input)

def Swish(input: Tensor) -> Tensor: 
    return SiLU(input)

def Softplus(input: Tensor) -> Tensor: 
    return F.log(1 + F.exp(input))

def Mish(input: Tensor) -> Tensor:
    return input * Tanh(Softplus(input))

def Hardtanh(
    input: Tensor, 
    min_value: float = -1.0, 
    max_value: float = 1.0
) -> Tensor:
    _full = lambda x : full(
        (), fill_value=x, dtype=input.dtype, device=input.device)
    return F.maximum(_full(min_value), F.minimum(_full(max_value), input))

def Hardsigmoid(input: Tensor) -> Tensor:
    _zeros = zeros((), dtype=input.dtype, device=input.device)
    _ones = ones((), dtype=input.dtype, device=input.device)
    return F.maximum(_zeros, F.minimum(_ones, (input + 1) / 2))

def Hardswish(input: Tensor) -> Tensor:
    return input * Hardsigmoid(input)

def Softsign(input: Tensor) -> Tensor:
    return input / (1 + input.abs())

def Softmin(input: Tensor) -> Tensor:
    return Softmax(-input)

