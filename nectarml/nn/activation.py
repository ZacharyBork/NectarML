from typing import Literal

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.nn.module import Module
import nectarml.functional as F

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

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
        return F.leaky_relu(x, self.negative_slope)

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
        return F.elu(x, self.alpha)
    
class SELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.selu(x)
    
class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)
    
class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)
    
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
        return F.softmax(x, self.dim)
    
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
        return F.log_softmax(x, self.dim)
    
class GeLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)
    
class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.swish(x)
    
class Softplus(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x)

class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.mish(x)
    
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
        return F.hardtanh(x, self.min_value, self.max_value)
    
class Hardsigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.hardsigmoid(x)
    
class Hardswish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.hardswish(x)
    
class Softsign(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softsign(x)
    
class Softmin(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softmin(x)

