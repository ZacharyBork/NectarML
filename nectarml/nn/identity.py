from __future__ import annotations

from typing import Any

from nectarml.core      import Tensor
from nectarml.nn.module import Module

class Identity(Module):
    def __init__(
        self:     Identity,
        *args:    list[Any],
        **kwargs: dict[str, Any]
    ) -> None:
        '''Identity layer. No-op, passes input through unchanged.
        
        Args:
            args   : Optional args for the identity module.
            kwargs : Optional kwargs for the identity module.
        '''
        super().__init__()
        
    def forward(self: Identity, x: Tensor) -> Tensor:
        return x
    
