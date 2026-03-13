from typing import Literal

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.creation import empty, zeros
import nectarml.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = empty(
            (self.out_features, self.in_features),
            dtype=self.dtype, device=self.device, requires_grad=True)
        nn.init.kaiming_normal_(
            weights=self.weights, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = zeros(
                (out_features,), dtype=dtype, 
                device=device, requires_grad=True)
        else: self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weights.transpose((1, 0))
        if self.bias is not None: y = self.bias + y
        return y
