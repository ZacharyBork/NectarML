from __future__ import annotations

from nectarml.tensor import Tensor
from nectarml.nn.module import Module
from nectarml.nn.init import kaiming_normal_
from nectarml.typing import DTypeLike, float32
from nectarml.creation import empty, zeros
from nectarml.amp.precision import amp_float16

class Linear(Module):
    def __init__(
        self: Linear,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(dtype)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = empty(
            (self.out_features, self.in_features),
            dtype=self.dtype, device='cpu', requires_grad=True)
        kaiming_normal_(
            weights=self.weights, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = zeros(
                (out_features,), dtype=dtype, 
                device='cpu', requires_grad=True)
        else: self.bias = None

    @amp_float16
    def forward(self: Linear, x: Tensor) -> Tensor:
        y = x @ self.weights.transpose((1, 0))
        if self.bias is not None: y = self.bias + y
        return y
