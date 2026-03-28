import warnings
from os import PathLike
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import numpy as np
from PIL import Image

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import full
from nectarml.typing import DTypeLike, float32
from nectarml.vision.transforms.transform import Transform
from nectarml.vision.transforms.format import ToTensor, ToPIL
from nectarml.vision.transforms.normalization import MinMaxNormalize
from nectarml.functional.interpolation import upsample

### IMAGE UTILS ###

class MakeGrid(Transform[Tensor | Sequence[Tensor], Tensor]):
    def __init__(
        self, 
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: tuple[int, int] = (0, 255),
        scale_each: bool = False,
        pad_value: float = 0.0
    ) -> None:
        super().__init__()
        self.nrow = nrow
        self.padding = padding
        self.value_range = value_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        
        if not normalize: self.norm = None
        else: self.norm = MinMaxNormalize(value_range[0], value_range[1])
        
    def forward(self, input: Tensor | Sequence[Tensor]) -> Tensor:
        if isinstance(input, Sequence): input = F.cat(input, dim=0)
        
        if self.scale_each:
            split = F.unbind(input, dim=0)
            if self.norm is not None: split = [self.norm(i) for i in split]
        else:
            if self.norm is not None: input = self.norm(input)
            split = F.unbind(input, dim=0)
        split = [i.unsqueeze(dim=0) for i in split]
                        
        count = len(split)
        rows = int(np.ceil(count / self.nrow))
        cols = int(np.minimum(count, self.nrow))
        size_h = split[0].shape[-2] + (self.padding * 2)
        size_w = split[0].shape[-1] + (self.padding * 2)
                
        canvas = full(
            (1, 3, size_h * rows, size_w * cols), 
            fill_value=self.pad_value)
        canvas = canvas.to(split[0].device, split[0].dtype)
            
        curr_row = curr_col = 0
        for i in range(count):        
            start = (
                size_h * curr_row + self.padding, 
                size_w * curr_col + self.padding)
            end = (
                size_h * (curr_row+1) - self.padding, 
                size_w * (curr_col+1) - self.padding)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                canvas[:, :, start[0]:end[0], start[1]:end[1]] = split[i]
            
            if curr_col > cols - 2:
                curr_col = 0
                curr_row += 1
            else: curr_col += 1
        
        return canvas

class LoadImageFile(Transform[None, Tensor]):
    def __init__(
        self, 
        image_path: PathLike,
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
    ) -> None:
        super().__init__(device)
        self.image_path = Path(image_path)
        self.to_tensor = ToTensor(device, dtype)
    
    def forward(self, *args, **kwargs) -> Tensor:
        assert self.image_path.exists(), (
            f'Unable to locate image file at path: '
            f'{self.image_path.as_posix()}')
        
        image = Image.open(self.image_path)
        return self.to_tensor(image)
    
    def __call__(self) -> Tensor:
        return self.forward()
    
class SaveImageFile(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        output_path: PathLike,
        normalize: bool = False
    ) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.normalize = normalize
        self.to_pil = ToPIL(normalize, value_range=(0, 255))
        
    def forward(self, input: Tensor) -> Tensor:
        out_dir = self.output_path.parent.resolve()
        assert out_dir.exists(), \
            f'Unable to locate output directory at path: {out_dir.as_posix()}'
        
        self.to_pil(input).save(self.output_path)
        return input
    
class Resample(Transform[Tensor, Tensor]):
    def __init__(
        self,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        a: float = -0.75,
        align_corners: bool = False
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.a = a
        self.align_corners = align_corners
        
    def forward(self, input: Tensor) -> Tensor:
        return upsample(input, self.size, self.scale_factor, self.mode,
            self.a, self.align_corners)

### SHAPE UTILS ###

class Permute(Transform[Tensor, Tensor]):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims
    
    def forward(self, input: Tensor) -> Tensor:
        return input.permute(self.dims)
    
class Transpose(Transform[Tensor, Tensor]):
    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, input: Tensor) -> Tensor:
        return input.transpose(self.dim1, self.dim2)
    
### VALUE UTILS ###
    
class Clamp(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        min_value: float | None = None,
        max_value: float | None = None
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, input: Tensor) -> Tensor:
        return input.clamp(self.min_value, self.max_value)
    
class MaskedFill(Transform[Tensor, Tensor]):
    def __init__(self, mask: Tensor, value: float = 0.0) -> None:
        super().__init__()
        self.mask = mask
        self.value = value
    
    def forward(self, input: Tensor) -> Tensor:
        return input.masked_fill(self.mask, self.value)

