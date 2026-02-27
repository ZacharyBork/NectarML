import random
from typing import Literal
from collections.abc import Sequence

from PIL import Image, ImageOps
import numpy as np

from nectarml.vision.transforms import Transform

class Pad(Transform):
    def __init__(
        self,
        padding: int | tuple[int, ...],
        fill: int | tuple[int, int, int] = 0,
        padding_mode: Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant'
    ) -> None:
        self.padding: tuple[int, ...] = None
        self.is_grayscale = False
        self._init_padding(padding)
        self._init_fill(fill)
        
        match padding_mode:
            case 'constant': self.op = self._constant
            case 'edge': self.op = self._edge
            case 'reflect': self.op = self._reflect
            case 'symmetric': self.op = self._symmetric
       
    ### PROPERTIES ###
    
    @property
    def np_pad_width(self) -> tuple[tuple[int, int], ...]:
        p = self.padding
        width = ((p[1], p[3]), (p[0], p[2]))
        if not self.is_grayscale: width += ((0, 0),)
        return width
    
    ### UTILS ###
        
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
                    
    def _init_fill(self, fill: int | tuple[int, int, int]) -> None:
        if isinstance(fill, int): self.fill = (fill,) * 3
        else: self.fill = fill
    
    def _pad_numpy(
        self, 
        input: Image.Image, 
        mode: Literal['edge', 'reflect', 'symmetric']
    ) -> Image.Image:
        array = np.array(input)
        self.is_grayscale = array.ndim == 2
        padded = np.pad(array, self.np_pad_width, mode=mode)
        return Image.fromarray(padded)
    
    ### PADDING MODES ###
    
    def _constant(self, input: Image.Image) -> Image.Image:
        return ImageOps.expand(input, self.padding, self.fill)
    
    def _edge(self, input: Image.Image) -> Image.Image:
        return self._pad_numpy(input, mode='edge')
        
    def _reflect(self, input: Image.Image) -> Image.Image:
        return self._pad_numpy(input, mode='reflect')
    
    def _symmetric(self, input: Image.Image) -> Image.Image:
        return self._pad_numpy(input, mode='symmetric')
    
    ### RUN ###
    
    def run(self, input: Image.Image) -> Image.Image:
        return self.op(input)

class RandomCrop(Transform):
    def __init__(
        self,
        size: int | tuple[int, int],
        padding: int | tuple[int,int] | tuple[int,int,int,int] | None = None,
        pad_if_needed: bool = False,
        fill: float | tuple[float, float, float] = 0.0,
        padding_mode: Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant'
    ) -> None:
        if isinstance(size, int): self.size = (size, size)
        else: self.size = size
        
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        if padding is not None: 
            self.pad = Pad(padding, fill, padding_mode)
        else: self.pad = None
    
    def _validate_input_size(self, input: Image.Image) -> Image.Image:
        if input.height < self.size[0] or input.width < self.size[1]:
            if self.pad_if_needed:
                diff_h = int(np.maximum(0, self.size[0] - input.height))
                diff_w = int(np.maximum(0, self.size[1] - input.width))
                padding = (diff_w, diff_h, diff_w, diff_h)
                self.pad = Pad(padding, self.fill, self.padding_mode)
                return self.pad(input)
            else:
                raise RuntimeError(
                    f'Input image size {input.size} is greater than '
                    f'desired crop size: {self.size}')
    
    def run(self, input: Image.Image) -> Image.Image:
        if self.pad is None: out = self._validate_input_size(input)
        else: out = self.pad(input)

        max_offset = (input.height - self.size[0], input.width - self.size[1])
        offset_h = int(random.random() * max_offset[0])
        offset_w = int(random.random() * max_offset[1])
        
        return out.crop((
            offset_w, offset_h, 
            offset_w + self.size[1], 
            offset_h + self.size[0]))
        
class CenterCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass
    
class RandomResizedCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass
    
class Resize(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomHorizontalFlip(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomVerticalFlip(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomRotation(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomAffine(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomPerspective(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class ElasticTransform(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class GridDistortion(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class OpticalDistortion(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class FiveCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class TenCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomCropNearBBox(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

