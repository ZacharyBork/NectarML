import warnings
from os import PathLike
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import numpy as np
from PIL import Image

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import full, zeros_like, ones_like, linspace
from nectarml.typing import DTypeLike, float32
from nectarml.vision.transforms.transform import \
    Transform, UtilityTransform, TransformInput
from nectarml.vision.transforms.format import ToTensor, ToPIL
from nectarml.vision.transforms.normalization import MinMaxNormalize
from nectarml.functional.interpolation import upsample

### DEBUG ###

class DebugPrint(Transform):
    def __init__(
        self, 
        op_name: str,
        shapes:  bool = True,
        dtypes:  bool = True,
        devices: bool = True,
        minmax:  bool = True
    ) -> None:
        super().__init__()
        self.op_name = op_name
        self.shapes  = shapes
        self.dtypes  = dtypes
        self.devices = devices
        self.minmax  = minmax
        
    def forward(self, input: TransformInput) -> TransformInput:
        output = f'{self.op_name}: ['
        input_dict = input.as_dict()
        for key, value in input_dict.items():
            output += f'\n    {key}: ['
            if value is None: 
                output += 'None]'
                continue
            output += f'\n        Type: {type(value)}'
            
            if isinstance(value, np.ndarray | Tensor):
                if self.shapes:
                    output += f',\n        Shape: {value.shape}'
                if self.dtypes:
                    output += f',\n        DType: {value.dtype}'
                if self.devices:
                    output += f',\n        Shape: {value.device}'
                if self.minmax:
                    _min, _max = value.min().item(), value.max().item()
                    output += f',\n        Min/Max: [{_min}, {_max}]'
            output += '\n    ]'
        
        output += '\n]'
        print(output)
        
        return input

### IMAGE UTILS ###

class MakeGrid(UtilityTransform[Tensor | Sequence[Tensor], Tensor]):
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

class LoadImageFile(UtilityTransform[None, Tensor]):
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
    
class SaveImageFile(UtilityTransform[Tensor, Tensor]):
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
    
class Resample(Transform):
    def __init__(
        self,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        a: float = -0.75,
        align_corners: bool = False,
        preserve_aspect_ratio: bool = False,
        transform_mask: bool = True
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.a = a
        self.align_corners = align_corners
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.transform_mask = transform_mask
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return upsample(input, self.size, self.scale_factor, self.mode,
            self.a, self.align_corners, self.preserve_aspect_ratio)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

### EVALUATION ###

class Derivative(Transform):
    def __init__(
        self,
        mode: Literal['ddx', 'ddy'] = 'ddx',
        per_channel: bool = False
    ) -> None:
        super().__init__()
        self.mode = mode
        self.per_channel = per_channel

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        if not self.per_channel:
            t = input.mean(dim=1, keepdim=True)
        else: t = input
        
        B, C, H, W = t.shape
        axis = 0 if self.mode == 'ddx' else 1
        outputs = []
        
        for b in range(B):
            channels = []
            for c in range(C):        
                if axis == 0: 
                    interior = (t[b, c, 2:, :]  - t[b, c, :-2, :]) / 2
                    left     =  t[b, c, 1:2, :] - t[b, c, 0:1, :]
                    right    =  t[b, c, -1:, :] - t[b, c, -2:-1, :]

                elif axis == 1:
                    interior = (t[b, c, :, 2:]  - t[b, c, :, :-2]) / 2
                    left     =  t[b, c, :, 1:2] - t[b, c, :, 0:1]
                    right    =  t[b, c, :, -1:] - t[b, c, :, -2:-1]
                 
                result = F.cat([left, interior, right], dim=axis).unsqueeze(0)
                channels.append(result)
            
            if self.per_channel: outputs.append(F.cat(channels, dim=0))
            else: outputs.extend(channels)
            
        return F.stack(outputs, dim=0)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class UVMap(Transform):
    def __init__(
        self,
        tiling_x: float = 1.0,
        tiling_y: float = 1.0
    ) -> None:
        super().__init__()
        self.tiling_x = tiling_x
        self.tiling_y = tiling_y
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape

        r = linspace(0, self.tiling_x, W, input.dtype, input.device)
        r = r.reshape((1, W)).expand((H, W))

        g = linspace(self.tiling_y, 0, H, input.dtype, input.device)
        g = g.reshape((H, 1)).expand((H, W))
        
        return F.stack([r, g, zeros_like(r)], dim=0).unsqueeze(0)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class NormalMap(Transform):
    def __init__(
        self,
        normal_power: float = 1.0,
        invert: bool = False
    ) -> None:
        super().__init__()
        assert normal_power > 0.0, '"normal_power" must be greater than 0.0.'
        self.normal_power = normal_power
        self.invert = invert

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        gray = input.mean(dim=1, keepdim=True)
        if self.invert: gray = 1.0 - gray
        dzdx = Derivative('ddx', per_channel=True)(gray)
        dzdy = Derivative('ddy', per_channel=True)(gray)
                
        normal = F.cat([-dzdx, -dzdy, ones_like(gray)], dim=1)
        length = (
            normal[:, 0, :, :]**2 
          + normal[:, 1, :, :]**2 
          + normal[:, 2, :, :]**2
        ).sqrt()

        return normal**(1/self.normal_power) / length * 0.5 + 0.5
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

### SHAPE UTILS ###

class Permute(Transform):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.permute(self.dims)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class Transpose(Transform):
    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.transpose(self.dim1, self.dim2)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
### VALUE UTILS ###
    
class Clamp(Transform):
    def __init__(
        self, 
        min_value: float | None = None,
        max_value: float | None = None
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.clamp(self.min_value, self.max_value)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class MaskedFill(Transform):
    def __init__(self, mask: Tensor, value: float = 0.0) -> None:
        super().__init__()
        self.mask = mask
        self.value = value
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.masked_fill(self.mask, self.value)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

