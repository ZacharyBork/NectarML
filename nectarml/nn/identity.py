from __future__ import annotations

from nectarml.nn.module import Module
from nectarml        import typing
from nectarml.core import Tensor

class Identity(Module):
    def __init__(
        self:  Identity,
        dtype: typing.dtype = typing.float32,
        *args,
        **kwargs
    ) -> None:
        super().__init__(dtype)
        
    def forward(self: Identity, x: Tensor) -> Tensor:
        return x
    
