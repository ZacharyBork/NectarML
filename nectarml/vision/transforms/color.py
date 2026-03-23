import random
import colorsys
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

import _nectarml
import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import full
from nectarml.vision.transforms import Transform
from nectarml.cuda.utils import map_dtype

### UTILS ###

def _hsv_adjust(
    input: Tensor,
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    value: float = 1.0,
    max_value: int | float | None = None
) -> Tensor:
    max_value = max_value or input.max().item()
    input = input / max_value
    if input.device == 'cuda':
        out_data = _nectarml.hsv_adjust(
            input._data_ptr, list(input.shape),
            hue_shift, saturation, value, map_dtype(input.dtype))
    else:
        img_array = (input.data).transpose((0, 2, 3, 1))
        
        hsv = np.vectorize(colorsys.rgb_to_hsv)(
            img_array[..., 0], img_array[..., 1], img_array[..., 2])
        
        h = (hsv[0] + hue_shift) % 1.0
        s = np.clip(hsv[1] * saturation, 0.0, 1.0)
        v = np.clip(hsv[2] * value, 0.0, 1.0)

        rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)
        out_data = np.clip(np.stack(rgb, axis=-1), 0, 255)
        out_data = out_data.transpose((0, 3, 1, 2)).astype(input.dtype)
        
    return Tensor(
        out_data, input.shape, input.dtype, input.device, input.requires_grad
    ) * max_value

### TRANSFORMS ###

class ColorJitter(Transform[Tensor, Tensor]):
    def __init__(
        self,
        brightness: float | tuple[float, float] = (0.9, 1.1),
        contrast: float | tuple[float, float] = (0.9, 1.1),
        saturation: float | tuple[float, float] = (0.9, 1.1),
        hue: float | tuple[float, float] = (-0.1, 0.1)
    ) -> None:
        super().__init__()
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
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        if not np.allclose(list(self.contrast), [1, 1]):
            _constrast = self._random_in_range(self.contrast)
            input = input / max_value
            input = ((input - 0.5) * _constrast + 0.5) * max_value
            input = input.clamp(0.0, max_value)

        _hue = self._random_in_range(self.hue)
        _sat = self._random_in_range(self.saturation)
        _val = self._random_in_range(self.brightness)
        out = _hsv_adjust(input, _hue, _sat, _val, max_value)
        
        return out
        
class RandomBrightness(Transform[Tensor, Tensor]):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.value_range = value_range
        self.p = p
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        brightness = self._random_in_range(self.value_range)
        return _hsv_adjust(input, 0.0, 1.0, brightness)

class RandomContrast(Transform[Tensor, Tensor]):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.value_range = value_range
        self.p = p
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance or np.allclose(list(self.value_range), [1, 1]): 
            return input
        
        max_value = input.max().item()
        out = input / max_value
        constrast = self._random_in_range(self.value_range)
        out = ((out - 0.5) * constrast + 0.5) * max_value
        return out.clamp(0.0, max_value)

class RandomSaturation(Transform[Tensor, Tensor]):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.value_range = value_range
        self.p = p
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        saturation = self._random_in_range(self.value_range)
        return _hsv_adjust(input, 0.0, saturation, 1.0)
    
class RandomHue(Transform[Tensor, Tensor]):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.value_range = value_range
        self.p = p
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        hue = self._random_in_range(self.value_range)
        return _hsv_adjust(input, hue, 1.0, 1.0)

class RandomGamma(Transform[Tensor, Tensor]):
    def __init__(
        self,
        value_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.value_range = value_range
        self.p = p
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance or np.allclose(list(self.value_range), [1, 1]): 
            return input
        
        max_value = input.max().item()
        gamma = self._random_in_range(self.value_range)
        out = (input / max_value) ** gamma * max_value
        return out.clamp(0.0, max_value)

class ToGrayscale(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        ch = input.unbind(dim=1)
        out = 0.2999 * ch[0] + 0.587 * ch[1] + 0.114 * ch[2]
        out = out.unsqueeze(dim=0).expand(input.shape)
        return out.to(input.device, input.dtype)

class RandomGrayscale(Transform[Tensor, Tensor]):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.to_grayscale = ToGrayscale()
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        return self.to_grayscale(input)

class ToSepia(Transform[Tensor, Tensor]):
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        in_r, in_g, in_b = input.unbind(dim=1)
        r = (in_r * 0.393) + (in_g * 0.769) + (in_b * 0.189)
        g = (in_r * 0.349) + (in_g * 0.686) + (in_b * 0.168)
        b = (in_r * 0.272) + (in_g * 0.534) + (in_b * 0.131)
        out = F.stack([r, g, b], dim=1).clamp(0.0, max_value)
        return out.to(input.device, input.dtype)
    
class RandomSepia(Transform[Tensor, Tensor]):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.to_sepia = ToSepia()
        
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        return self.to_sepia(input)

class Equalize(Transform[Tensor, Tensor]):
    # NOTE: Equalize always happens on CPU regardless of input Tensor's device.
    
    def __init__(
        self,
        mode: Literal['cv2', 'pil'] = 'pil',
        by_channel: bool = True
    ) -> None:
        super().__init__()
        self.mode = mode
        self.by_channel = by_channel
    
    def _eq_cv(self, input: Tensor) -> np.ndarray:
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

        return np.concatenate(outputs, axis=0)
    
    def _eq_pil(self, input: Tensor) -> np.ndarray:
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
        
        return np.concatenate(outputs, axis=0)
    
    def forward(self, input: Tensor) -> Tensor:
        match self.mode.strip().casefold():
            case 'cv2': out_data = self._eq_cv(input)
            case 'pil': out_data = self._eq_pil(input)
            case _: raise ValueError(f'Invalid Equalize mode: {self.mode}')

        return Tensor(
            out_data, input.shape, input.dtype, 
            input.device, input.requires_grad)

class AutoContrast(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Solarize(Transform[Tensor, Tensor]):
    '''
    Reference:
        - https://msameeruddin.hashnode.dev/solarizing-the-image-with-numpy
    '''
    def __init__(
        self,
        threshold_range: tuple[float, float] = (0.3, 0.7),
        per_channel: bool = False
    ) -> None:
        super().__init__()
        self.threshold_range = threshold_range
        self.per_channel = per_channel
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        
        if self.per_channel:
            thresholds = [
                self._random_in_range(self.threshold_range)
                for _ in range(3)]
        else: thresholds = [self._random_in_range(self.threshold_range)] * 3
        
        r, g, b = (input / max_value).unbind(dim=1)
        channels = [
            F.where((r < thresholds[0]), r, 1-r),
            F.where((g < thresholds[1]), g, 1-g),
            F.where((b < thresholds[2]), b, 1-b)]
        return (F.stack(channels, dim=1) * max_value).clamp(0.0, max_value)

class Posterize(Transform[Tensor, Tensor]):
    def __init__(self, levels: int = 10) -> None:
        super().__init__()
        assert levels >= 2, 'levels must be >= 2'
        self.levels = levels

    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        step = max_value / self.levels
        return (input / step).floor() * step

class Invert(Transform[Tensor, Tensor]):
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        base = full(input.shape, max_value, input.dtype).to(input.device)
        return (base - input).clamp(0.0, max_value)

class CLAHE(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ChannelShuffle(Transform[Tensor, Tensor]):
    def forward(self, input: Tensor) -> Tensor:
        channels = input.unbind(dim=1)
        random.shuffle(channels)
        return F.stack(channels, dim=1).to(input.device, input.dtype)

class ChannelDropout(Transform[Tensor, Tensor]):
    def __init__(
        self,
        range: tuple[int, int] = (1, 1),
        fill: float = 0.0
    ) -> None:
        super().__init__()
        self.range = range
        self.fill = fill
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        channels = input.unbind(dim=1)
        
        new_channel = full(channels[0].shape, self.fill) * max_value
        new_channel = new_channel.to(input.device, input.dtype)
        
        index = random.randint(self.range[0], self.range[1])
        channels[index] = new_channel
        return F.stack(channels, dim=1).clamp(0.0, max_value)

class RGBShift(Transform[Tensor, Tensor]):
    def __init__(
        self,
        r_shift_limit: tuple[int, int] = (-20, 20),
        g_shift_limit: tuple[int, int] = (-20, 20),
        b_shift_limit: tuple[int, int] = (-20, 20)
    ) -> None:
        super().__init__()
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        remapped = input / max_value * 255.0
        r, g, b = remapped.unbind(dim=1)
        
        r += self._random_in_range(self.r_shift_limit)
        g += self._random_in_range(self.g_shift_limit)
        b += self._random_in_range(self.b_shift_limit)
        
        channels = [r, g, b]
        out = F.stack(channels, dim=1)
        out = (out / 255.0 * max_value).clamp(0.0, max_value)
        return out
    
class HueSaturationValue(Transform[Tensor, Tensor]):
    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        value: float = 1.0
    ) -> None:
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
    
    def forward(self, input: Tensor) -> Tensor:
        return _hsv_adjust(input, self.hue, self.saturation, self.value)

class TonemapHDR(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

