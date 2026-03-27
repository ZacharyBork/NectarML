from __future__ import annotations

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.nn.module import Module

class ReLU(Module):
    def __init__(self: ReLU, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: ReLU, x: Tensor) -> Tensor:
        if self.inplace: return F.relu_(x)
        return F.relu(x)

class LeakyReLU(Module):
    def __init__(
        self: LeakyReLU,
        negative_slope: float = 0.01,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self: LeakyReLU, x: Tensor) -> Tensor:
        if self.inplace: return F.leaky_relu_(x, self.negative_slope)
        return F.leaky_relu(x, self.negative_slope)

class ELU(Module):
    def __init__(
        self: ELU,
        alpha: float = 1.0,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self: ELU, x: Tensor) -> Tensor:
        if self.inplace: return F.elu_(x, self.alpha)
        return F.elu(x, self.alpha)
    
class SELU(Module):
    def __init__(self: SELU, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: SELU, x: Tensor) -> Tensor:
        if self.inplace: return F.selu_(x)
        return F.selu(x)
    
class Sigmoid(Module):
    def __init__(self: Sigmoid, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Sigmoid, x: Tensor) -> Tensor:
        if self.inplace: return F.sigmoid_(x)
        return F.sigmoid(x)
    
class Tanh(Module):
    def __init__(self: Tanh, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Tanh, x: Tensor) -> Tensor:
        if self.inplace: return F.tanh_(x)
        return F.tanh(x)
    
class Softmax(Module):
    def __init__(
        self: Softmax,
        dim: int = -1,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        
    def forward(self: Softmax, x: Tensor) -> Tensor:
        if self.inplace: return F.softmax_(x, self.dim)
        return F.softmax(x, self.dim)
    
class LogSoftmax(Module):
    def __init__(
        self: LogSoftmax,
        dim: int = -1,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        
    def forward(self: LogSoftmax, x: Tensor) -> Tensor:
        if self.inplace: return F.log_softmax_(x, self.dim)
        return F.log_softmax(x, self.dim)
    
class GeLU(Module):
    def __init__(self: GeLU, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: GeLU, x: Tensor) -> Tensor:
        if self.inplace: return F.gelu_(x)
        return F.gelu(x)

class SiLU(Module):
    def __init__(self: SiLU, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: SiLU, x: Tensor) -> Tensor:
        if self.inplace: return F.silu_(x)
        return F.silu(x)
    
class Swish(Module):
    def __init__(self: Swish, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Swish, x: Tensor) -> Tensor:
        if self.inplace: return F.swish_(x)
        return F.swish(x)
    
class Softplus(Module):
    def __init__(self: Softplus, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Softplus, x: Tensor) -> Tensor:
        if self.inplace: return F.softplus_(x)
        return F.softplus(x)

class Mish(Module):
    def __init__(self: Mish, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Mish, x: Tensor) -> Tensor:
        if self.inplace: return F.mish_(x)
        return F.mish(x)
    
class Hardtanh(Module):
    def __init__(
        self: Hardtanh,
        min_value: float = -1.0, 
        max_value: float = 1.0,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.inplace = inplace
        
    def forward(self: Hardtanh, x: Tensor) -> Tensor:
        if self.inplace: return F.hardtanh_(x, self.min_value, self.max_value)
        return F.hardtanh(x, self.min_value, self.max_value)
    
class Hardsigmoid(Module):
    def __init__(self: Hardsigmoid, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Hardsigmoid, x: Tensor) -> Tensor:
        if self.inplace: return F.hardsigmoid_(x)
        return F.hardsigmoid(x)
    
class Hardswish(Module):
    def __init__(self: Hardswish, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Hardswish, x: Tensor) -> Tensor:
        if self.inplace: return F.hardswish_(x)
        return F.hardswish(x)
    
class Softsign(Module):
    def __init__(self: Softsign, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Softsign, x: Tensor) -> Tensor:
        if self.inplace: return F.softsign_(x)
        return F.softsign(x)
    
class Softmin(Module):
    def __init__(self: Softmin, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self: Softmin, x: Tensor) -> Tensor:
        if self.inplace: return F.softmin_(x)
        return F.softmin(x)

