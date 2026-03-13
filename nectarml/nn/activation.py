from typing import Literal

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.nn.module import Module
import nectarml.functional as F

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.ReLU(x)

class LeakyReLU(Module):
    def __init__(
        self,
        negative_slope: float = 0.01,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.negative_slope = negative_slope
    
    def forward(self, x: Tensor) -> Tensor:
        return F.LeakyReLU(x, self.negative_slope)

class ELU(Module):
    def __init__(
        self,
        alpha: float = 1.0,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        return F.ELU(x, self.alpha)
    
class SELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.SELU(x)
    
class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Sigmoid(x)
    
class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Tanh(x)
    
class Softmax(Module):
    def __init__(
        self,
        dim: int = -1,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return F.Softmax(x, self.dim)
    
class LogSoftmax(Module):
    def __init__(
        self,
        dim: int = -1,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return F.LogSoftmax(x, self.dim)
    
class GeLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.GeLU(x)

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.SiLU(x)
    
class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Swish(x)
    
class Softplus(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Softplus(x)

class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Mish(x)
    
class Hardtanh(Module):
    def __init__(
        self,
        min_value: float = -1.0, 
        max_value: float = 1.0,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, x: Tensor) -> Tensor:
        return F.Hardtanh(x, self.min_value, self.max_value)
    
class Hardsigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Hardsigmoid(x)
    
class Hardswish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Hardswish(x)
    
class Softsign(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Softsign(x)
    
class Softmin(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Softmin(x)

