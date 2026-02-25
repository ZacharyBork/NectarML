from typing import Literal

from nectarml import Tensor, DTypeLike, float32
import nectarml.nn as nn

class Identity(nn.Module):
    def __init__(
        self,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32,
        *args,
        **kwargs
    ) -> None:
        super().__init__(device, dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        return x
    
