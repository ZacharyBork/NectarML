from __future__ import annotations

from nectarml.core      import Tensor
from nectarml.nn.module import Module

class Identity(Module):
    def __init__(
        self:  Identity,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        
    def forward(self: Identity, x: Tensor) -> Tensor:
        return x
    
