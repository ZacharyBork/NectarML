import random
from typing import Literal
from collections.abc import Sequence

from PIL import Image, ImageOps
import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

class Pad(Transform):
    def __init__(
        self,
        padding: int | tuple[int, ...],
        fill: float = 0.0,
        padding_mode: Literal[
            'constant', 'reflect', 'replicate', 'circular'
        ] = 'constant',
    ) -> None:
        super().__init__()
        self.padding: tuple[int, ...] = None
        self.fill = fill
        self.padding_mode = padding_mode
        
        self._init_padding(padding)
                
    def _init_padding(self, padding: int | tuple[int, ...]) -> None:
        if isinstance(padding, int): self.padding = (padding,) * 4
        elif isinstance(padding, tuple):            
            if len(padding) == 2:
                self.padding = (padding[0], padding[1], padding[0], padding[1])
            elif len(padding) == 4: self.padding = padding
            else: 
                raise ValueError(
                    'RandomCrop padding tuple must have either 2 [LT, RB] or '
                    '4 [L, T, R, B] values.')
        else: raise ValueError('Pad.padding must be int or tuple[int, ...].')
        
    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, self.padding_mode, self.fill)
    
class RandomCrop(Transform):
    def __init__(
        self,
        size: int | tuple[int, int],
        padding: int | tuple[int, ...] | None = None,
        pad_if_needed: bool = False,
        fill: float = 0.0,
        padding_mode: Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant'
    ) -> None:
        super().__init__()
        if isinstance(size, int): self.size = (size, size)
        else: self.size = size
        
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        if padding is not None: self.pad = Pad(padding, fill, padding_mode)
        else: self.pad = None
    
    def _validate_input_size(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        if H < self.size[0] or W < self.size[1]:
            if self.pad_if_needed:
                diff_h = int(np.maximum(0, self.size[0] - H))
                diff_w = int(np.maximum(0, self.size[1] - W))
                padding = (diff_w, diff_h, diff_w, diff_h)
                self.pad = Pad(padding, self.fill, self.padding_mode)
                return self.pad(input)
            else:
                raise RuntimeError(
                    f'Input image size {input.shape[2:]} is greater than '
                    f'desired crop size: {self.size}')
        else: return input
    
    def forward(self, input: Tensor) -> Tensor:
        if self.pad is None: out = self._validate_input_size(input)
        else: out = self.pad(input)

        max_offset = (input.shape[2]-self.size[0], input.shape[3]-self.size[1])
        offset_h = int(random.random() * max_offset[0])
        offset_w = int(random.random() * max_offset[1])
        
        return out[
            :, :, 
            offset_h:offset_h+self.size[0], 
            offset_w:offset_w+self.size[1]]
        
class CenterCrop(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass
    
class RandomResizedCrop(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass
    
class Resize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomHorizontalFlip(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomVerticalFlip(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomRotation(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomAffine(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomPerspective(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ElasticTransform(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class GridDistortion(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class OpticalDistortion(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class FiveCrop(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class TenCrop(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomCropNearBBox(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

