from typing import Any, Literal, TypeVar, Generic
from collections.abc import Iterable

import numpy as np

from nectarml.tensor import Tensor
from nectarml.random import RNG

TInputType  = TypeVar('TInputType')
TOutputType = TypeVar('TOutputType')

class Transform(Generic[TInputType, TOutputType]):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] | None = None
    ) -> None:
        self.device = device
        self.rng = RNG
    
    ### UTILS ###
    
    def _random_in_range(
        self, 
        value_range: tuple[int | float, int | float] = (0.0, 1.0)
    ) -> float:
        self.rng.randfloat(value_range[0], value_range[1])
        
    ### FORWARD ###
    
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, input: TInputType) -> TOutputType:
        return self.forward(input)
    
    ### INSPECTION ###
    
    def __repr__(self) -> str:
        return f'{self.__class__}'
    

