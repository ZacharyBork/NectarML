import warnings
from os import PathLike
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import grey_erosion, grey_dilation

import _nectarml
import nectarml.functional as F
from nectarml import typing
from nectarml.tensor import Tensor
from nectarml.creation import full, zeros_like, ones_like, linspace
from nectarml.functional.interpolation import upsample

from nectarml.vision.transforms.transform import Transform, UtilityTransform
from nectarml.vision.transforms.common import TransformInput
from nectarml.vision.transforms.format import ToTensor, ToPIL
from nectarml.vision.transforms.normalization import MinMaxNormalize

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

class NoOp(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: TransformInput) -> TransformInput:
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
        dtype: typing.dtype = typing.float32,
        device: typing.DeviceLikeType = 'cpu',
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
        
        B, C, _, _ = t.shape
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
        
class ApplyLUT(Transform):
    def __init__(
        self,
        lut_file: PathLike,
        alpha: float = 1.0,
        p: float = 1.0
    ) -> None:
        super().__init__(p=p)
        lut_file = Path(lut_file)
        assert lut_file.exists, \
            f'Unable to locate LUT file at: {lut_file.as_posix()}'
        assert lut_file.suffix == '.cube', 'LUTs must be ".cube" files.'
        self._read_lut(lut_file)
        self.alpha = alpha
        
    def _read_lut(self, lut_file: Path) -> None:
        with open(lut_file, 'r') as file:
            data = file.readlines()
        read_data = False
        lut_data = []
        for i in data:
            if 'LUT_3D_SIZE' in i:
                self.lut_size = int(i.replace('\n', '').split(' ')[-1])
                read_data = True
                continue
            if read_data:
                if i.strip() == '': continue
                values = i.strip().split()
                if len(values) == 3:
                    lut_data.append([float(x) for x in values])
        
        S = self.lut_size
        arr = np.array(lut_data, dtype=np.float32).reshape(S, S, S, 3)
        arr = arr.transpose(2, 1, 0, 3)
        self.lut = Tensor(arr, dtype=typing.float32)
            
    def _apply_lut_cpu(self, image: np.ndarray) -> np.ndarray:
        S = self.lut_size

        r_idx = image[:, 0] * (S - 1)
        g_idx = image[:, 1] * (S - 1)
        b_idx = image[:, 2] * (S - 1)

        r0 = np.clip(r_idx.astype(np.int32), 0, S - 1)
        g0 = np.clip(g_idx.astype(np.int32), 0, S - 1)
        b0 = np.clip(b_idx.astype(np.int32), 0, S - 1)
        r1 = np.clip(r0 + 1, 0, S - 1)
        g1 = np.clip(g0 + 1, 0, S - 1)
        b1 = np.clip(b0 + 1, 0, S - 1)

        rf = (r_idx - r0)[..., np.newaxis]
        gf = (g_idx - g0)[..., np.newaxis]
        bf = (b_idx - b0)[..., np.newaxis]

        lut = self.lut.numpy()

        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]

        c00 = c000 + rf * (c100 - c000)
        c01 = c001 + rf * (c101 - c001)
        c10 = c010 + rf * (c110 - c010)
        c11 = c011 + rf * (c111 - c011)
        c0  = c00  + gf * (c10  - c00)
        c1  = c01  + gf * (c11  - c01)
        out = c0   + bf * (c1   - c0)

        return out.transpose(0, 3, 1, 2).astype(np.float32)
      
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        B, C, H, W = input.shape
        assert C == 3, 'LUT application requires RGB input'
        
        max_value = input.max().item()
        norm = input / max_value
        self.lut = self.lut.to(input.device)
        
        if input.device == 'cuda':
            out_data = _nectarml.apply_lut(
                norm._data_ptr, self.lut._data_ptr,
                B, H, W, self.lut_size,
                norm.dtype.cuda)
        else: out_data = self._apply_lut_cpu(input.numpy())
            
        out = Tensor(out_data, input.shape, input.dtype, input.device)
        return (out * max_value).clamp(0.0, max_value)

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
        self.mask = self.mask.to(input.device, input.dtype)
        return input.masked_fill(self.mask, self.value)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class Morphological(Transform):
    def __init__(
        self,
        scale: int | tuple[int, int] = (13, 15),
        operation: Literal['dilation', 'erosion'] = 'dilation',
        per_channel: bool = False,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.scale = (scale, scale) if isinstance(scale, int) else scale
        self.operation = operation
        self.per_channel = per_channel

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

        if not self.per_channel:
            arr = input.numpy()
            if self.operation == 'dilation':
                output = grey_dilation(arr, size=self._scale)
            else: output = grey_erosion(arr, size=self._scale)
        else:
            channels = input.unbind(dim=1)
            outputs = []
            for ch in channels:
                arr = ch.numpy()
                if self.operation == 'dilation':
                    outputs.append(grey_dilation(arr, size=self._scale))
                else: outputs.append(grey_erosion(arr, size=self._scale))
            output = np.stack(outputs, axis=1)
                  
        return Tensor(output, dtype=input.dtype, device=input.device)

    def _build_parameters(self) -> None:
        self._scale = int(self._random_in_range(self.scale))

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

### OVERLAY ###

class OverlayElements(Transform):
    def __init__(
        self,
        element: Tensor,
        location: float | tuple[float, float] = (0.1, 0.1),
        scale: float | tuple[float, float] = (0.2, 0.2),
        resample_mode: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear',
        preserve_aspect_ratio: bool = True,
        p: float = 1.0
    ) -> None:
        '''
        - Pivot point is top left corner of element.
        - Reference location is from top left corner of input.
        - Scale is taken as percentage of input.
        '''
        super().__init__(p=p)
        self.element = element
        self.location = (location, location) \
            if isinstance(location, int | float) else location
        self.scale = (scale, scale) \
            if isinstance(scale, int | float) else scale
        self.resample_mode = resample_mode
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        output = input.clone()
        
        element = self.element.to(input.device, input.dtype)
        B = input.shape[0]
        for b in range(B):
            _, H, W = input[b].shape
            
            size = (int(H * self.scale[0]), int(W * self.scale[1]))
            scaled_element = upsample(
                element, size=size, mode=self.resample_mode,
                preserve_aspect_ratio=self.preserve_aspect_ratio)
                        
            start_y = int(self.location[0] * H)
            end_y = int(self.location[0] * H) + scaled_element.shape[-2]
            
            start_x = int(self.location[1] * W)
            end_x = int(self.location[1] * W) + scaled_element.shape[-1]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output[b, :, start_y:end_y, start_x:end_x] = scaled_element
            
        return output
            
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
     
class OverlayText(Transform):
    def __init__(
        self,
        text: str,
        font: str | None = None,
        font_size: int = 36,
        text_color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int] | None = None,
        background_padding: int = 5,
        location: float | tuple[float, float] = (0.1, 0.1),
        p: float = 1.0
    ) -> None:
        super().__init__(p=p)
        self.text = text
        self.font = ImageFont.truetype(font, font_size) \
            if font is not None else ImageFont.load_default(font_size)
        self.font_size = font_size
        self.text_color = text_color
        self.background_color = background_color
        self.background_padding = (
            -background_padding, -background_padding,
            background_padding, background_padding) \
            if background_color is not None else (0, 0, 0, 0)
        self.location = (location, location) \
            if isinstance(location, int | float) else location

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        output = input.clone()
        im_col = self.background_color \
              if self.background_color is not None \
            else (0, 0, 0) if self.text_color != (0, 0, 0) \
            else (255, 255, 255)
        
        B = input.shape[0]
        for b in range(B):
            _, H, W = input[b].shape
            
            image = Image.new('RGB', size=(H, W), color=im_col)
            draw = ImageDraw.Draw(image)

            draw_loc = (self.background_padding[2], self.background_padding[3])
            bbox = draw.textbbox(draw_loc, self.text, self.font)
            bbox = tuple([x+y for x, y in zip(bbox, self.background_padding)])
            draw.text(draw_loc, self.text, fill=self.text_color, font=self.font)
            image = image.crop(bbox)

            arr = np.array(image).transpose(2, 0, 1)[np.newaxis]
            arr = arr.astype(input.dtype.numpy) / 255
            text = Tensor(arr, dtype=input.dtype, device=input.device)

            start_y = int(self.location[0] * H)
            end_y = int(self.location[0] * H) + text.shape[-2]
            
            start_x = int(self.location[1] * W)
            end_x = int(self.location[1] * W) + text.shape[-1]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.background_color is not None:
                    output[b, :, start_y:end_y, start_x:end_x] = text
                else:
                    op = F.maximum if self.text_color != (0,0,0) else F.minimum
                    orig = output[b, :, start_y:end_y, start_x:end_x]
                    output[b, :, start_y:end_y, start_x:end_x] = op(orig, text)

        return output
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

