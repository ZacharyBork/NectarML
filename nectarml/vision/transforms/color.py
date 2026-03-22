import random
import colorsys
from PIL import Image, ImageEnhance
import numpy as np

import _nectarml
import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import full
from nectarml.vision.transforms import Transform
from nectarml.benchmark import benchmark_time
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

class ColorJitter(Transform):
    def __init__(
        self,
        brightness: float | tuple[float, float] = (0.9, 1.1),
        contrast: float | tuple[float, float] = (0.9, 1.1),
        saturation: float | tuple[float, float] = (0.9, 1.1),
        hue: float | tuple[float, float] = (-0.1, 0.1)
    ) -> None:
        super().__init__()
        for parameter in [brightness, contrast, saturation]:
            if isinstance(parameter, float): 
                parameter = (1.0 - parameter, parameter)
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
        
class RandomBrightness(Transform):
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

class RandomContrast(Transform):
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

class RandomSaturation(Transform):
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
    
class RandomHue(Transform):
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

class RandomGamma(Transform):
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

class ToGrayscale(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        ch = input.unbind(dim=1)
        out = 0.2999 * ch[0] + 0.587 * ch[1] + 0.114 * ch[2]
        out = out.unsqueeze(dim=0).expand(input.shape)
        return out.to(input.device, input.dtype)

class RandomGrayscale(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.to_grayscale = ToGrayscale()
    
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        return self.to_grayscale(input)

class ToSepia(Transform):
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        in_r, in_g, in_b = input.unbind(dim=1)
        r = (in_r * 0.393) + (in_g * 0.769) + (in_b * 0.189)
        g = (in_r * 0.349) + (in_g * 0.686) + (in_b * 0.168)
        b = (in_r * 0.272) + (in_g * 0.534) + (in_b * 0.131)
        out = F.stack([r, g, b], dim=1).clamp(0.0, max_value)
        return out.to(input.device, input.dtype)
    
class RandomSepia(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.to_sepia = ToSepia()
        
    def forward(self, input: Tensor) -> Tensor:
        chance = self._random_in_range()
        if self.p <= chance: return input
        return self.to_sepia(input)

class Equalize(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class AutoContrast(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Solarize(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Posterize(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Invert(Transform):
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        base = full(input.shape, max_value, input.dtype).to(input.device)
        return (base - input).clamp(0.0, max_value)

class CLAHE(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ChannelShuffle(Transform):
    def forward(self, input: Tensor) -> Tensor:
        channels = input.unbind(dim=1)
        random.shuffle(channels)
        return F.stack(channels, dim=1).to(input.device, input.dtype)

class ChannelDropout(Transform):
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

class RGBShift(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class HueSaturationValue(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class TonemapHDR(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

