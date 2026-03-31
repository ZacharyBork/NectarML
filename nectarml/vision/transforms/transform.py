from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar, Generic

from nectarml.tensor import Tensor
from nectarml.random import RNG

TInputType  = TypeVar('TInputType')
TOutputType = TypeVar('TOutputType')

@dataclass
class TransformInput:
    image:     Tensor
    image2:    Tensor | None = None
    mask:      Tensor | None = None
    boxes:     Tensor | None = None
    keypoints: Tensor | None = None

    def __post_init__(self):
        assert self.image is not None, \
            'TransformInput requires at least an image.'

    @classmethod
    def from_args(
        cls,
        args: tuple[Tensor, ...],
        kwargs: dict[str, Tensor]
    ) -> TransformInput:
        '''
        Positional Convention:
        (image,)
        (image, mask)
        (image, image2, mask)
        '''
        if kwargs: return cls(**kwargs)
        
        match len(args):
            case 1: return cls(image=args[0])
            case 2: return cls(image=args[0], mask=args[1])
            case 3: return cls(image=args[0], image2=args[1], mask=args[2])
            case _: 
                raise ValueError(
                    f'Expected 1-3 positional args (image, image2, mask) '
                    f'or keyword args, got {len(args)}')

    def to_output(
        self,
        original_args: tuple,
        original_kwargs: dict
    ) -> Tensor | tuple[Tensor, ...]:
        if original_kwargs:
            return tuple(
                {k: getattr(self, k) for k in original_kwargs}.values())

        match len(original_args):
            case 1: return self.image
            case 2: return self.image, self.mask
            case 3: return self.image, self.image2, self.mask
            
    def as_dict(self) -> dict[str, Tensor | None]:
        return {
            'image':     self.image,
            'image2':    self.image2,
            'mask':      self.mask,
            'boxes':     self.boxes,
            'keypoints': self.keypoints
        }
            
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
        return self.rng.randfloat(value_range[0], value_range[1])
        
    ### FORWARD ###
    
    def forward(self, input: TransformInput) -> TransformInput:
        raise NotImplementedError
    
    def __call__(
        self, 
        *args: Tensor, 
        **kwargs: Tensor
    ) -> Tensor | tuple[Tensor]:
        input = TransformInput.from_args(args, kwargs)
        result = self.forward(input)
        return result.to_output(args, kwargs)
    
    ### INSPECTION ###
    
    def __repr__(self) -> str:
        return f'{self.__class__}'
    
class UtilityTransform(Transform, Generic[TInputType, TOutputType]):
    def __init__(
        self, 
        device: Literal['cpu', 'cuda'] | None = None
    ) -> None:
        super().__init__(device)
        
    def __call__(self, input: TInputType) -> TOutputType:
        return self.forward(input)
    
    

