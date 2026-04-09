import math
import warnings
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import Size, float32
from nectarml.creation import full, ones, zeros, linspace
from nectarml.vision.transforms.transform import Transform 
from nectarml.vision.transforms.spatial import OpticalDistortion
from nectarml.vision.transforms.common import \
    TransformInput, hsv_adjust, gradient_mask, lerp3

### TRANSFORMS ###

class ColorJitter(Transform):
    def __init__(
        self,
        brightness: float | tuple[float, float] = (0.9, 1.1),
        contrast: float | tuple[float, float] = (0.9, 1.1),
        saturation: float | tuple[float, float] = (0.9, 1.1),
        hue: float | tuple[float, float] = (-0.1, 0.1),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        if isinstance(brightness, float): 
            brightness = (1.0 - brightness, 1.0 + brightness)
        if isinstance(contrast, float): 
            contrast = (1.0 - contrast, 1.0 + contrast)
        if isinstance(saturation, float): 
            saturation = (1.0 - saturation, 1.0 + saturation)
        if isinstance(hue, float): hue = (-hue, hue)
        
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        
        if not np.allclose(list(self.contrast), [1, 1]):
            input = input / max_value
            input = ((input - 0.5) * self._contrast + 0.5) * max_value
            input = input.clamp(0.0, max_value)

        return hsv_adjust(input, self._hue, self._sat, self._val, max_value)
    
    def _build_parameters(self) -> None:
        self._hue      = self._random_in_range(self.hue)
        self._sat      = self._random_in_range(self.saturation)
        self._val      = self._random_in_range(self.brightness)
        self._contrast = self._random_in_range(self.contrast)
    
    def forward(self, input: TransformInput) -> TransformInput:        
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class RandomBrightness(Transform):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.value_range = value_range
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return hsv_adjust(input,  0.0, 1.0, self._brightness)
    
    def _build_parameters(self) -> None:
        self._brightness = self._random_in_range(self.value_range)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomContrast(Transform):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.value_range = value_range
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        out = (((input / max_value) - 0.5) * self._contrast + 0.5) * max_value
        return out.clamp(0.0, max_value)
    
    def _build_parameters(self) -> None:
        self._contrast = self._random_in_range(self.value_range)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomSaturation(Transform):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.value_range = value_range
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return hsv_adjust(input,  0.0, self._saturation, 1.0)
    
    def _build_parameters(self) -> None:
        self._saturation = self._random_in_range(self.value_range)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomHue(Transform):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.value_range = value_range
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return hsv_adjust(input, self._hue, 1.0, 1.0)
    
    def _build_parameters(self) -> None:
        self._hue = self._random_in_range(self.value_range)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomGamma(Transform):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.value_range = value_range
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        out = (input / max_value) ** self._gamma * max_value
        return out.clamp(0.0, max_value)
    
    def _build_parameters(self) -> None:
        self._gamma = self._random_in_range(self.value_range)
    
    def forward(self, input: TransformInput) -> TransformInput:        
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToGrayscale(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        ch = input.unbind(dim=1)
        out = 0.2999 * ch[0] + 0.587 * ch[1] + 0.114 * ch[2]
        out = out.unsqueeze(dim=0).expand(input.shape)
        return out.to(input.device, input.dtype)
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToBlackAndWhite(Transform):
    def __init__(
        self,
        white_point: int = 125,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.white_point = white_point / 255
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        gray = norm.mean(dim=1, keepdim=True)
        result = F.where(
            (gray > self.white_point), 
            ones((), input.dtype, input.device), 0.0)

        return result * max_value
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToSepia(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        in_r, in_g, in_b = input.unbind(dim=1)
        r = (in_r * 0.393) + (in_g * 0.769) + (in_b * 0.189)
        g = (in_r * 0.349) + (in_g * 0.686) + (in_b * 0.168)
        b = (in_r * 0.272) + (in_g * 0.534) + (in_b * 0.131)
        out = F.stack([r, g, b], dim=1).clamp(0.0, max_value)
        return out.to(input.device, input.dtype)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Equalize(Transform):
    # NOTE: Equalize always happens on CPU regardless of input Tensor's device.
    
    def __init__(
        self,
        mode: Literal['cv2', 'pil'] = 'pil',
        by_channel: bool = True,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.mode = mode
        self.by_channel = by_channel
    
    def _eq_cv(self, input: Tensor) -> Tensor:
        batches = input.unbind(dim=0)
        outputs: list[np.ndarray] = []
        
        for batch in batches:
            if not self.by_channel:
                data = batch.permute((1, 2, 0))
                img_yuv = cv2.cvtColor(data, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                out_data = np.array(img_output).astype(input.dtype)
                out_data = np.ascontiguousarray(out_data.transpose((2, 0, 1)))
            else:
                channels = batch.unbind(dim=0)
                arrs = []
                for ch in channels:
                    equalized = cv2.equalizeHist(ch.numpy().astype(np.uint8))
                    arrs.append(np.array(equalized).astype(input.dtype))
                out_data = np.ascontiguousarray(np.stack(arrs, axis=0))
            
            outputs.append(out_data)

        return Tensor(
            np.concatenate(outputs, axis=0), input.shape, input.dtype, 
            input.device, input.requires_grad)
    
    def _eq_pil(self, input: Tensor) -> Tensor:
        batches = input.unbind(dim=0)
        outputs: list[np.ndarray] = []
        
        for batch in batches:
            if not self.by_channel:
                out = batch.permute((1, 2, 0))
                img = Image.fromarray(
                    out.numpy().astype(dtype=np.uint8), 'RGB')
                img = ImageOps.equalize(img)
                out_data = np.array(img).astype(input.dtype)
                out_data = np.ascontiguousarray(out_data.transpose((2, 0, 1)))
            else:
                channels = batch.unbind(dim=0)
                arrs = []
                for ch in channels:
                    img = Image.fromarray(
                        ch.numpy().astype(dtype=np.uint8), 'L')
                    img = ImageOps.equalize(img)
                    arrs.append(np.array(img).astype(input.dtype))
                out_data = np.ascontiguousarray(np.stack(arrs, axis=0))
                
            outputs.append(out_data)
        
        return Tensor(
            np.concatenate(outputs, axis=0), input.shape, input.dtype, 
            input.device, input.requires_grad)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        match self.mode.strip().casefold():
            case 'cv2': return self._eq_cv(input)
            case 'pil': return self._eq_pil(input)
            case _: raise ValueError(f'Invalid Equalize mode: {self.mode}')
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class AutoContrast(Transform):
    def __init__(
        self, 
        cutoff: float = 0.0,
        method: Literal['cdf', 'pil'] = 'pil',
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        assert 0.0 <= cutoff <= 1.0, \
            'AutoContrast cutoff should be in 0-1 range.'
        self.cutoff = cutoff * 100.0
        self.method = method
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        if self.cutoff == 0.0:
            lo = norm.amin(dim=(-2, -1), keepdim=True)
            hi = norm.amax(dim=(-2, -1), keepdim=True)
        else:
            B, C, H, W = norm.shape
            lo_list, hi_list = [], []
            for c in range(C):
                channel = norm[:, c]
                lo = F.quantile(channel, self.cutoff / 100.0)
                lo_list.append(lo.reshape((1, 1, 1, 1)))
                hi = F.quantile(channel, 1 - self.cutoff / 100.0)
                hi_list.append(hi.reshape((1, 1, 1, 1)))
            lo = F.cat(lo_list, dim=1)
            hi = F.cat(hi_list, dim=1)

        return ((norm - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0) * max_value
    
    def forward(self, input: Tensor) -> Tensor:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class Solarize(Transform):
    '''
    Reference:
        - https://msameeruddin.hashnode.dev/solarizing-the-image-with-numpy
    '''
    def __init__(
        self,
        threshold_range: tuple[float, float] = (0.3, 0.7),
        per_channel: bool = False,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.threshold_range = threshold_range
        self.per_channel = per_channel
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        r, g, b = (input / max_value).unbind(dim=1)
        channels = [
            F.where((r < self._thresholds[0]), r, 1-r),
            F.where((g < self._thresholds[1]), g, 1-g),
            F.where((b < self._thresholds[2]), b, 1-b)]
        return (F.stack(channels, dim=1) * max_value).clamp(0.0, max_value)
    
    def _build_parameters(self) -> None:
        if self.per_channel:
            self._thresholds = [
                self._random_in_range(self.threshold_range)
                for _ in range(3)]
        else: self._thresholds = [
            self._random_in_range(self.threshold_range)] * 3
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class Posterize(Transform):
    def __init__(self, levels: int = 10, p: float = 0.5) -> None:
        super().__init__(p=p)
        assert levels >= 2, 'levels must be >= 2'
        self.levels = levels

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        step = max_value / self.levels
        return (input / step).floor() * step

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class Invert(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        base = full(input.shape, max_value, input.dtype).to(input.device)
        return (base - input).clamp(0.0, max_value)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class CLAHE(Transform):
    def __init__(
        self,
        clip_limit: int = 4,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = (tile_grid_size, tile_grid_size) \
            if isinstance(tile_grid_size, int) else tile_grid_size
        
    def _clahe_mapping(self, tile: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))
        
        limit = self.clip_limit * tile.size / 256
        excess = np.sum(np.maximum(hist - limit, 0))
        hist = np.minimum(hist, limit)
        hist += excess / 256
        
        cdf = np.cumsum(hist)
        cdf = (cdf - cdf.min()) / (tile.size - cdf.min()) * 255
        return cdf.astype(np.float32)        

    def _interp(
        self, 
        input: np.ndarray, 
        mappings: np.ndarray, 
        th: int, 
        tw: int, 
        ty: int, 
        tx: int
    ) -> np.ndarray:
        H, W = input.shape
        result = np.zeros_like(input, dtype=np.float32)
        ys, xs = np.mgrid[0:H, 0:W]

        ty_f = ys / th - 0.5
        tx_f = xs / tw - 0.5
        
        t_row0 = np.clip(np.floor(ty_f).astype(int), 0, ty - 1)
        t_col0 = np.clip(np.floor(tx_f).astype(int), 0, tx - 1)
        t_row1 = np.clip(t_row0 + 1, 0, ty - 1)
        t_col1 = np.clip(t_col0 + 1, 0, tx - 1)
        
        wy1 = np.clip(ty_f - t_row0, 0.0, 1.0)
        wx1 = np.clip(tx_f - t_col0, 0.0, 1.0)
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1
        
        v = input.astype(int)
        q00 = mappings[t_row0, t_col0, v]
        q10 = mappings[t_row1, t_col0, v]
        q01 = mappings[t_row0, t_col1, v]
        q11 = mappings[t_row1, t_col1, v]
        
        result = (
            wy0 * wx0 * q00 
          + wy1 * wx0 * q10 
          + wy0 * wx1 * q01 
          + wy1 * wx1 * q11)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        B, C, H, W = input.shape
        ty, tx = self.tile_grid_size
        max_val = input.max().item()

        arr = (input.cpu().numpy() / max_val * 255).astype(np.uint8)
        out = np.zeros_like(arr, dtype=np.float32)
        
        th, tw = H // ty, W // tx
    
        for b in range(B):
            for c in range(C):
                channel = arr[b, c]
                mappings = np.zeros((ty, tx, 256), dtype=np.float32)
                for row in range(ty):
                    for col in range(tx):
                        tile = channel[row*th:(row+1)*th, col*tw:(col+1)*tw]
                        mappings[row, col] = self._clahe_mapping(tile)
                
                out[b, c] = self._interp(channel, mappings, th, tw, ty, tx)
        
        out = (out / 255.0 * max_val).astype(input.dtype)
        return Tensor(out, dtype=input.dtype).to(input.device)
        
    def forward(self, input: Tensor) -> Tensor:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ChannelShuffle(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        ch = input.unbind(dim=1)
        shuffled = [ch[i] for i in self._channels]
        return F.stack(shuffled, dim=1).to(input.device, input.dtype)

    def _build_parameters(self, C: int) -> None:
        self._channels = list(range(C))
        self.rng.shuffle(self._channels)
        
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape[1])
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ChannelDropout(Transform):
    def __init__(
        self,
        range: tuple[int, int] = (1, 1),
        fill: float = 0.0,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.range = range
        self.fill = fill
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        channels = input.unbind(dim=1)
        
        new_channel = full(channels[0].shape, self.fill) * max_value
        channels[self._index] = new_channel.to(input.device, input.dtype)
        return F.stack(channels, dim=1).clamp(0.0, max_value)
    
    def _build_parameters(self) -> None:
        self._index = self.rng.randint(self.range[0], self.range[1])
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RGBShift(Transform):
    def __init__(
        self,
        r_shift_limit: tuple[int, int] = (-20, 20),
        g_shift_limit: tuple[int, int] = (-20, 20),
        b_shift_limit: tuple[int, int] = (-20, 20),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        remapped = input / max_value * 255.0
        r, g, b = remapped.unbind(dim=1)
        
        channels = [r + self._r, g + self._g, b + self._b]
        out = F.stack(channels, dim=1)
        return (out / 255.0 * max_value).clamp(0.0, max_value)
    
    def _build_parameters(self) -> None:
        self._r = self._random_in_range(self.r_shift_limit)
        self._g = self._random_in_range(self.g_shift_limit)
        self._b = self._random_in_range(self.b_shift_limit)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class HueSaturationValue(Transform):
    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        value: float = 1.0,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.hue = hue
        self.sat = saturation
        self.val = value
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return hsv_adjust(input, self.hue, self.sat, self.val)
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class TonemapHDR(Transform):
    def __init__(
        self, 
        method: Literal[
            'reinhard', 'exposure+gamma', 'filmic', 'aces'
        ] = 'reinhard',
        exposure: float = 1.0,
        gamma: float = 2.2,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.method = method
        self.exposure = exposure
        self.gamma = gamma
    
    def _hable(self, x: Tensor | float) -> Tensor:
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
    
    def _aces(self, x: Tensor | float) -> Tensor:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return (x * (a*x + b)) / (x * (c*x + d) + e)        
    
    def _gamma_encode(self, rgb: Tensor) -> Tensor:
        return rgb.clamp(0, 1) ** (1.0 / self.gamma)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        match self.method:
            case 'reinhard': 
                r, g, b = norm.unbind(dim=1)
                luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
                curve = luma / (1 + luma)
                tonemapped = norm * (curve / (luma + 1e-8))
            case 'exposure+gamma':
                r, g, b = norm.unbind(dim=1)
                luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
                curve = 1 - (-self.exposure * luma).exp()
                tonemapped = input * curve
            case 'filmic':
                curr = self._hable(norm * self.exposure)
                white = self._hable(11.2)
                tonemapped = curr / white
            case 'aces': 
                tonemapped = self._aces(norm * self.exposure).clamp(0, 1)
        
        return self._gamma_encode(tonemapped) * max_value
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ChromaticAberration(Transform):
    def __init__(
        self,
        inner_distortion_limit: float | tuple[float, float] = (-0.03, 0.03),
        outer_distortion_limit: float | tuple[float, float] = (-0.15, 0.15),
        falloff_power: float = 2.0,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        inner, outer = inner_distortion_limit, outer_distortion_limit
        self.inner_limit = (-inner, inner) \
            if isinstance(inner, float | int) else inner
        self.outer_limit = (-outer, outer) \
            if isinstance(outer, float | int) else outer
        self.falloff_power = falloff_power
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        _, _, H, W = input.shape
        channels = input.unbind(dim=1)
        inner = zeros(input.shape, input.dtype).to(input.device)
        outer = zeros(input.shape, input.dtype).to(input.device)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inner[:, 0, :, :] = self._inner1(channels[0].reshape((1, 1, H, W)))
            inner[:, 1, :, :] = channels[1]
            inner[:, 2, :, :] = self._inner2(channels[2].reshape((1, 1, H, W)))
            
            outer[:, 0, :, :] = self._outer1(channels[0].reshape((1, 1, H, W)))
            outer[:, 1, :, :] = channels[1]
            outer[:, 2, :, :] = self._outer1(channels[2].reshape((1, 1, H, W)))
        
        mask = gradient_mask(input.shape, 'radial', 'edges')
        mask = (mask**self.falloff_power).to(input.device, input.dtype)
        output = lerp3(input, inner, outer, mask)
        
        return output
        
    def _build_parameters(self) -> None:
        inner_limit = self._random_in_range(self.inner_limit)
        outer_limit = self._random_in_range(self.outer_limit) + inner_limit
        
        self._inner1 = OpticalDistortion(inner_limit, inner_limit, p=1)
        self._inner2 = OpticalDistortion(inner_limit/2, inner_limit/2, p=1)
        
        self._outer1 = OpticalDistortion(outer_limit, inner_limit, p=1)
        self._outer2 = OpticalDistortion(outer_limit/2, inner_limit/2, p=1)
        
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Vignetting(Transform):
    def __init__(
        self,
        intensity_range: float | tuple[float, float] = (0.2, 0.5),
        center_range: float | tuple[float, float] = (0.3, 0.7),
        p: float = 0.5    
    ) -> None:
        super().__init__(p=p)
        self.intensity_range = (intensity_range, intensity_range) \
            if isinstance(intensity_range, float | int) else intensity_range
        self.center_range = (center_range, center_range) \
            if isinstance(center_range, float | int) else center_range
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        mask = gradient_mask(input.shape, 'elliptical', 'corners')
        mask = mask.to(input.device, input.dtype)
        mask = 1.0 - (mask - self._center).clamp(0.0, 1.0)
        mask = mask ** (1.0 + self._intensity)
        output = input * mask
        
        return output.clamp(0.0, input.max().item())
        
    def _build_parameters(self) -> None:
        self._intensity = self._random_in_range(self.intensity_range)
        self._center = self._random_in_range(self.center_range)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Illumination(Transform):
    def __init__(
        self,
        mode: Literal['linear', 'radial'] = 'radial',
        intensity_range: float | tuple[float, float] = (0.1, 0.3),
        effect_type: Literal['brighten', 'darken', 'both'] = 'both',
        angle_range: tuple[float, float] = (0.0, 360.0),
        center_range: tuple[float, float] = (0.25, 0.75),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.mode = mode
        self.intensity_range = (intensity_range, intensity_range) \
            if isinstance(intensity_range, (int, float)) else intensity_range
        self.effect_type = effect_type
        self.angle_range = angle_range
        self.center_range = center_range

    def _linear_mask(self, H: int, W: int) -> Tensor:
        angle_rad = math.radians(self._angle)
        yy = linspace(0, H-1, H, dtype=float32).reshape((H, 1)).expand((H, W))
        xx = linspace(0, W-1, W, dtype=float32).reshape((1, W)).expand((H, W))
        proj = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        return proj.to(dtype=float32)

    def _radial_mask(self, H: int, W: int) -> Tensor:
        yy = linspace(0, H-1, H, dtype=float32).reshape((H, 1)).expand((H, W))
        xx = linspace(0, W-1, W, dtype=float32).reshape((1, W)).expand((H, W))
        dist = ((xx - self._cx)**2 + (yy - self._cy)**2).sqrt()
        dist = dist / (math.sqrt(2) + 1e-8)
        mask = 1.0 - dist
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.to(dtype=float32)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        max_value = input.max().item()

        if self.mode == 'linear': mask = self._linear_mask(H, W)
        else: mask = self._radial_mask(H, W)
        mask = mask.to(input.device, input.dtype)

        if self.effect_type == 'brighten': delta = mask * self._intensity
        elif self.effect_type == 'darken': delta = -mask * self._intensity
        else: delta = (mask - 0.5) * 2 * self._intensity

        return (input + (delta * max_value)).clamp(0.0, max_value)

    def _build_parameters(self) -> None:
        self._intensity = self._random_in_range(self.intensity_range)
        self._angle     = self._random_in_range(self.angle_range)
        self._cx        = self._random_in_range(self.center_range)
        self._cy        = self._random_in_range(self.center_range)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Quantize(Transform):
    def __init__(
        self,
        levels: int | tuple[int, int] = (3, 5),
        palette: Tensor | None = None,
        p: float = 0.5    
    ) -> None:
        super().__init__(p=p)
        self.levels = (levels, levels) if isinstance(levels, int) else levels
        for i in self.levels:
            assert i > 0, 'Quantize levels must be greater than 0.'
        self.palette = palette
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        if self.palette is None:
            return (input * self._levels).round() / self._levels
        else:
            palette = self.palette.to(input.device, input.dtype)
            diff = (input.unsqueeze(-1) - palette)
            idx = diff.abs().argmin(dim=-1)
            return palette[idx]
                
    def _build_parameters(self) -> None:
        self._levels = int(round(self._random_in_range(self.levels)))
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
