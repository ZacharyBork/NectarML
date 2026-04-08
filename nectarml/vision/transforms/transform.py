from __future__ import annotations

from typing import TypeVar, Generic

from nectarml.tensor import Tensor
from nectarml.random import RNG
from nectarml.vision.transforms.common import TransformInput

TInputType  = TypeVar('TInputType')
TOutputType = TypeVar('TOutputType')
            
class Transform():
    def __init__(self, p: float = 1.0) -> None:
        self.rng = RNG
        self.p = p
    
    ### UTILS ###
    
    def _random_in_range(
        self, 
        value_range: tuple[int | float, int | float] = (0.0, 1.0)
    ) -> float:
        return self.rng.randfloat(value_range[0], value_range[1])
        
    ### FORWARD ###
    
    def forward(self, input: TransformInput) -> TransformInput:
        raise NotImplementedError
    
    def _call(self, input: TransformInput) -> TransformInput:
        if self.rng.random() > self.p: result = input
        else: result = self.forward(input)
        return result
    
    def __call__(
        self, 
        *args: Tensor, 
        **kwargs: Tensor
    ) -> Tensor | tuple[Tensor]:
        input = TransformInput.from_args(args, kwargs)
        result = self._call(input)
        return result.to_output(args, kwargs)
    
    ### INSPECTION ###
    
    def __repr__(self) -> str:
        return f'{self.__class__}'
    
class UtilityTransform(Transform, Generic[TInputType, TOutputType]):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, input: TInputType) -> TOutputType:
        return self.forward(input)

