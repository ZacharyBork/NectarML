from __future__ import annotations

from nectarml           import typing
from nectarml.tensor    import Tensor
from nectarml.nn.module import Module
from nectarml.nn.init   import kaiming_normal_
from nectarml.creation  import empty, zeros

class Linear(Module):
    def __init__(
        self:         Linear,
        in_features:  int,
        out_features: int,
        bias:         bool  = True,
        dtype:        typing.dtype = typing.float32
    ) -> None:
        super().__init__(dtype)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = empty(
            (self.out_features, self.in_features),
            dtype=self.dtype, device='cpu', requires_grad=True)
        kaiming_normal_(
            weights=self.weight, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = zeros(
                (out_features,), dtype=dtype, 
                device='cpu', requires_grad=True)
        else: self.bias = None

    def forward(self: Linear, x: Tensor) -> Tensor:
        y = x @ self.weight.transpose(1, 0)
        if self.bias is not None: y = self.bias + y
        return y
