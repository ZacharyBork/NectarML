from __future__ import annotations

import builtins

import nectarml.nn.functional as F
from nectarml.core      import Tensor
from nectarml.nn.module import Module

class ReLU(Module):
    def __init__(self: ReLU, inplace: builtins.bool = False) -> None:
        '''Rectified linear unit activation function.
        
        Returns the input value if the value is >= 0.0, otherwise returns 0.0.
        
        Equation: f(x) = max(0, x)
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: ReLU, x: Tensor) -> Tensor:
        if self.inplace: return F.relu_(x)
        return F.relu(x)

class LeakyReLU(Module):
    def __init__(
        self:           LeakyReLU,
        negative_slope: builtins.float = 0.01,
        inplace:        builtins.bool  = False
    ) -> None:
        '''Leaky rectified linear unit activation function.
        
        Variant of ReLU which allows small non-zero gradients. Helps keep 
        neurons active even with very small gradients.
        
        Equation: f(x) = x if x > 0 else negative_slope * x
        
        Args:
            negative_slope : Defines how much variance should be preserved in
                             the negative region.
            inplace        : If True, the data of the input tensor will be 
                             swapped in-place with the resuling data of the 
                             activation function, rather than creating a new 
                             output tensor. Slightly reduces memory overhead.
        '''
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace        = inplace
    
    def forward(self: LeakyReLU, x: Tensor) -> Tensor:
        if self.inplace: return F.leaky_relu_(x, self.negative_slope)
        return F.leaky_relu(x, self.negative_slope)

class ELU(Module):
    def __init__(
        self:    ELU,
        alpha:   builtins.float = 1.0,
        inplace: builtins.bool  = False
    ) -> None:
        '''Exponential linear unit activation function.
        
        ReLU activation which applies an exponential curve to negative inputs
        rather than the linear curve of traditional ReLU. Can further help to 
        avoid dying gradients, especially in deep networks.
            
        Equation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
        
        Args:
            alpha   : Multiplier to control saturation level of negative 
                      inputs.
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.alpha   = alpha
        self.inplace = inplace
    
    def forward(self: ELU, x: Tensor) -> Tensor:
        if self.inplace: return F.elu_(x, self.alpha)
        return F.elu(x, self.alpha)
    
class SELU(Module):
    def __init__(self: SELU, inplace: builtins.bool = False) -> None:
        '''Scaled exponential linear unit activation function.
            
        Similar to ELU activation. SELU automatically normalizes gradients to a
        fixed constant to help stabilize training.
            
        Equation: f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: SELU, x: Tensor) -> Tensor:
        if self.inplace: return F.selu_(x)
        return F.selu(x)
    
class Sigmoid(Module):
    def __init__(self: Sigmoid, inplace: builtins.bool = False) -> None:
        '''Sigmoid activation function.
        
        Maps incoming gradients to an S-shaped curve where (0 <= x <= 1). 
        Useful when converting inputs to probabilities for classifier networks.
            
        Equation: f(x) = 1 / (1 + exp(-x))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Sigmoid, x: Tensor) -> Tensor:
        if self.inplace: return F.sigmoid_(x)
        return F.sigmoid(x)
    
class Tanh(Module):
    def __init__(self: Tanh, inplace: builtins.bool = False) -> None:
        '''Tanh activation function.
            
        Maps input values along a 0-centered hyperbolic tangent curve.
            
        Equation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Tanh, x: Tensor) -> Tensor:
        if self.inplace: return F.tanh_(x)
        return F.tanh(x)
    
class Softmax(Module):
    def __init__(
        self:    Softmax,
        dim:     builtins.int  = -1,
        inplace: builtins.bool = False
    ) -> None:
        '''Softmax activation function.
        
        Generally used in classification tasks. Converts raw output scores into
        [0:1] probabilities.
            
        Equation: f(x) = exp(x_i) / sum(exp(x_j))
        
        Args:
            dim     : The dimension along which to compute the activation 
                      function.
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.dim     = dim
        self.inplace = inplace
        
    def forward(self: Softmax, x: Tensor) -> Tensor:
        if self.inplace: return F.softmax_(x, self.dim)
        return F.softmax(x, self.dim)
    
class Softmin(Module):
    def __init__(self: Softmin, inplace: builtins.bool = False) -> None:
        '''Softmin activation function.
        
        Equation: f(x) = exp(-x_i) / sum(exp(-x_j))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Softmin, x: Tensor) -> Tensor:
        if self.inplace: return F.softmin_(x)
        return F.softmin(x)
    
class LogSoftmax(Module):
    def __init__(
        self:    LogSoftmax,
        dim:     builtins.int  = -1,
        inplace: builtins.bool = False
    ) -> None:
        '''Log softmax activation function.
        
        Computes the logarithm of Softmax(x). Offers improved stability over 
        the traditional Softmax activation.
            
        Equation: f(x) = log(exp(x_i) / sum(exp(x_j)))
        
        Args:
            dim     : The dimension along which to compute the activation 
                      function.
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.dim     = dim
        self.inplace = inplace
        
    def forward(self: LogSoftmax, x: Tensor) -> Tensor:
        if self.inplace: return F.log_softmax_(x, self.dim)
        return F.log_softmax(x, self.dim)
    
class GeLU(Module):
    def __init__(self: GeLU, inplace: builtins.bool = False) -> None:
        '''Approximate gaussian error linear unit activation function.
        
        Smooth approximation of ReLU which scales inputs by the cumulative 
        distribution function of a Gaussian distribution. Enables better 
        gradient flow and helps to reduce vanishing gradients in deep networks, 
        and especially in transformer based networks.
        
        Equation: f(x) = x*0.5 * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
        
        Ref: https://arxiv.org/pdf/1606.08415
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: GeLU, x: Tensor) -> Tensor:
        if self.inplace: return F.gelu_(x)
        return F.gelu(x)

class SiLU(Module):
    def __init__(self: SiLU, inplace: builtins.bool = False) -> None:
        '''Sigmoid-weighted linear unit activation function.
        
        Similar to ReLU, SiLU (or Swish) is a smooth, self-gated activation 
        which multiplies inputs by their own Sigmoid transformation. Allows for 
        more precise weight accuracy in deep networks and helps to reduce the 
        effects of vanishing gradients by allowing small negative weights.
            
        Equation: f(x) = x * (1 / (1 + exp(-x)))
        
        Ref: https://arxiv.org/pdf/1710.05941v1
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: SiLU, x: Tensor) -> Tensor:
        if self.inplace: return F.silu_(x)
        return F.silu(x)
    
class Swish(Module):
    def __init__(self: Swish, inplace: builtins.bool = False) -> None:
        '''Sigmoid-weighted linear unit activation function.
            
        Similar to ReLU, SiLU (or Swish) is a smooth, self-gated activation 
        which multiplies inputs by their own Sigmoid transformation. Allows for 
        more precise weight accuracy in deep networks and helps to reduce the 
        effects of vanishing gradients by allowing small negative weights.
            
        Equation: f(x) = x * (1 / (1 + exp(-x)))
        
        Ref: https://arxiv.org/pdf/1710.05941v1
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Swish, x: Tensor) -> Tensor:
        if self.inplace: return F.swish_(x)
        return F.swish(x)
    
class Softplus(Module):
    def __init__(self: Softplus, inplace: builtins.bool = False) -> None:
        '''Softplus activation function.
            
        Equation: f(x) = log(1 + exp(x))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Softplus, x: Tensor) -> Tensor:
        if self.inplace: return F.softplus_(x)
        return F.softplus(x)

class Mish(Module):
    def __init__(self: Mish, inplace: builtins.bool = False) -> None:
        '''Mish activation function.
            
        Equation: f(x) = x * tanh(log(1 + exp(x)))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Mish, x: Tensor) -> Tensor:
        if self.inplace: return F.mish_(x)
        return F.mish(x)
    
class Hardtanh(Module):
    def __init__(
        self:      Hardtanh,
        min_value: builtins.float = -1.0, 
        max_value: builtins.float = 1.0,
        inplace:   builtins.bool  = False
    ) -> None:
        '''Hardtanh activation function.
        
        Equation: f(x) = max(min_value, min(max_value, x))
        
        Args:
            min_value : The minimum allowable value. Values below this will be
                clamped to this value.
            max_value : The maximum allowable value. Values above this will be
                clamped to this value.
            inplace   : If True, the data of the input tensor will be swapped 
                        in-place with the resuling data of the activation
                        function, rather than creating a new output tensor. 
                        Slightly reduces memory overhead.
        '''
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.inplace   = inplace
        
    def forward(self: Hardtanh, x: Tensor) -> Tensor:
        if self.inplace: return F.hardtanh_(x, self.min_value, self.max_value)
        return F.hardtanh(x, self.min_value, self.max_value)
    
class Hardsigmoid(Module):
    def __init__(self: Hardsigmoid, inplace: builtins.bool = False) -> None:
        '''Hardsigmoid activation function.
        
        Equation: f(x) = clamp((x + 1) / 2, 0, 1)
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Hardsigmoid, x: Tensor) -> Tensor:
        if self.inplace: return F.hardsigmoid_(x)
        return F.hardsigmoid(x)
    
class Hardswish(Module):
    def __init__(self: Hardswish, inplace: builtins.bool = False) -> None:
        '''Hardswish activation function.
        
        Equation: f(x) = x * max(0, min(1, (x + 1) / 2))
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Hardswish, x: Tensor) -> Tensor:
        if self.inplace: return F.hardswish_(x)
        return F.hardswish(x)
    
class Softsign(Module):
    def __init__(self: Softsign, inplace: builtins.bool = False) -> None:
        '''Softsign activation function.
            
        Equation: f(x) = x / (1 + |x|)
        
        Args:
            inplace : If True, the data of the input tensor will be swapped in-
                      place with the resuling data of the activation function,
                      rather than creating a new output tensor. Slightly
                      reduces memory overhead.
        '''
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Softsign, x: Tensor) -> Tensor:
        if self.inplace: return F.softsign_(x)
        return F.softsign(x)


