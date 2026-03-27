from __future__ import annotations

from nectarml.nn import Module
from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32

class Identity(Module):
    def __init__(
        self: Identity,
        dtype: DTypeLike = float32,
        *args,
        **kwargs
    ) -> None:
        super().__init__(dtype)
        
    def forward(self: Identity, x: Tensor) -> Tensor:
        return x
    
