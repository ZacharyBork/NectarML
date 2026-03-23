from typing import Literal, TypeVar, Generic

import numpy as np

from nectarml.tensor import Tensor

TInputType  = TypeVar('TInputType')
TOutputType = TypeVar('TOutputType')

class Transform(Generic[TInputType, TOutputType]):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] | None = None
    ) -> None:
        self.device = device
        self.rng = np.random.default_rng()
    
    ### UTILS ###
    
    def _random_in_range(
        self, 
        value_range: tuple[int | float, int | float] = (0.0, 1.0)
    ) -> float:
        _min = value_range[0]
        _max = value_range[1]
        value = _min + (_max - _min) * self.rng.random() 
        return value
    
    ### FORWARD ###
    
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, input: TInputType) -> TOutputType:
        return self.forward(input)
    
    ### INSPECTION ###
    
    def __repr__(self) -> str:
        return f'{self.__class__}'
    

