from nectarml.tensor import Tensor
from nectarml.nn.module import Module
import nectarml.functional as F

class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.relu_(x)
        return F.relu(x)

class LeakyReLU(Module):
    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.leaky_relu_(x, self.negative_slope)
        return F.leaky_relu(x, self.negative_slope)

class ELU(Module):
    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.elu_(x, self.alpha)
        return F.elu(x, self.alpha)
    
class SELU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.selu_(x)
        return F.selu(x)
    
class Sigmoid(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.sigmoid_(x)
        return F.sigmoid(x)
    
class Tanh(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.tanh_(x)
        return F.tanh(x)
    
class Softmax(Module):
    def __init__(
        self,
        dim: int = -1,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.softmax_(x, self.dim)
        return F.softmax(x, self.dim)
    
class LogSoftmax(Module):
    def __init__(
        self,
        dim: int = -1,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.log_softmax_(x, self.dim)
        return F.log_softmax(x, self.dim)
    
class GeLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.gelu_(x)
        return F.gelu(x)

class SiLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.silu_(x)
        return F.silu(x)
    
class Swish(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.swish_(x)
        return F.swish(x)
    
class Softplus(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.softplus_(x)
        return F.softplus(x)

class Mish(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.mish_(x)
        return F.mish(x)
    
class Hardtanh(Module):
    def __init__(
        self,
        min_value: float = -1.0, 
        max_value: float = 1.0,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.hardtanh_(x, self.min_value, self.max_value)
        return F.hardtanh(x, self.min_value, self.max_value)
    
class Hardsigmoid(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.hardsigmoid_(x)
        return F.hardsigmoid(x)
    
class Hardswish(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.hardswish_(x)
        return F.hardswish(x)
    
class Softsign(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.softsign_(x)
        return F.softsign(x)
    
class Softmin(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace: return F.softmin_(x)
        return F.softmin(x)

