from __future__ import annotations

from typing import Literal

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.random import RNG
            
class Generator():
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        self.size   = size
        self.dtype  = dtype
        self.device = device
        self.rng    = RNG
    
    ### FORWARD ###
    
    def forward(self) -> Tensor:
        raise NotImplementedError
    
    def generate(self) -> Tensor:
        result = self.forward()
        return result.to(self.device, self.dtype)
    
    ### INSPECTION ###
    
    def __repr__(self) -> str:
        return f'{self.__class__}'
    

