from typing import Any, Literal, TypeVar, Generic
from collections.abc import Iterable

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
    
    def _random_index(
        self, 
        min_index: int = 0,
        max_index: int = 5
    ) -> float:
        return int(round(self._random_in_range((min_index, max_index))))
    
    def _random_selection(self, items: Iterable) -> Any:
        items = list(items)
        index = self._random_index(max_index=len(items))
        return items[index]
    
    ### FORWARD ###
    
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, input: TInputType) -> TOutputType:
        return self.forward(input)
    
    ### INSPECTION ###
    
    def __repr__(self) -> str:
        return f'{self.__class__}'
    

