from typing import Any, Literal, TypeVar, Generic

from PIL import Image
import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision import utils
from nectarml.typing import uint8

TInputType  = TypeVar('TInputType')
TOutputType = TypeVar('TOutputType')

class Transform(Generic[TInputType, TOutputType]):
    def __init__(
        self, 
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        self.device = device
        self.original_type: Literal['image', 'ndarray', 'tensor'] = None
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
    

