from __future__ import annotations

from nectarml        import typing
from nectarml.tensor import Tensor
from nectarml.random import RNG
            
class Generator():
    def __init__(
        self, 
        size:   tuple[int, int] = (256, 256),
        dtype:  typing.dtype = typing.float32,
        device: typing.DeviceLikeType = 'cpu'
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
    

