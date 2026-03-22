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

def random_hue_shift(input: Tensor) -> Tensor:
    if input.device == 'cuda':
        pass
    else: pass
    
def random_brightness(input: Tensor) -> Tensor:
    if input.device == 'cuda':
        pass
    else: pass
    
def random_contrast(input: Tensor) -> Tensor:
    if input.device == 'cuda':
        pass
    else: pass
    
def random_color(input: Tensor) -> Tensor:
    if input.device == 'cuda':
        pass
    else: pass

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
    
    def _hue_shift(self, input: Image.Image) -> Image.Image:
        shift = (self._random_in_range(self.hue) * 360.0 % 360)
        arr = np.array(input.convert("RGB"), dtype=np.uint8)
        with benchmark_time():
            result = _nectarml.hue_shift(arr, shift)
        return Image.fromarray(result)
    
        
        # shift = (self._random_in_range(self.hue) * 360.0 % 360)
        # img_array = np.array(input.convert("RGB"), dtype=np.float32) / 255.0
        
        # with benchmark_time():
        #     hsv = np.vectorize(colorsys.rgb_to_hsv)(
        #         img_array[..., 0], img_array[..., 1], img_array[..., 2])
        #     shifted_h = (hsv[0] + shift) % 1.0
            
        #     rgb = np.vectorize(colorsys.hsv_to_rgb)(shifted_h, hsv[1], hsv[2])
        #     result = (np.stack(rgb, axis=-1) * 255).astype(np.uint8)
        
        # return Image.fromarray(result)
    
    def forward(self, input: Tensor) -> Tensor:
        if not self.brightness == (1.0, 1.0):
            input = ImageEnhance.Brightness(input).enhance(
                self._random_in_range(self.brightness))
        if not self.contrast == (1.0, 1.0):
            input = ImageEnhance.Contrast(input).enhance(
                self._random_in_range(self.contrast))
        if not self.saturation == (1.0, 1.0):
            input = ImageEnhance.Color(input).enhance(
                self._random_in_range(self.saturation))
        if not self.hue == (0.0, 0.0):
            input = self._hue_shift(input)
        return input
        
class RandomBrightness(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomContrast(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomSaturation(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass
    
class RandomHue(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomGamma(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomGrayscale(Transform):
    def __init__(
        self,
        chance: tuple[float, float] = (0.05, 0.2)
    ) -> None:
        super().__init__()
        self.chance = chance
    
    def forward(self, input: Tensor) -> Tensor:
        rand = self.rng.random()
        chance = rand * (self.chance[1]-self.chance[0]) + self.chance[0]
        value = self.rng.random()
        if value <= chance: 
            ch = input.unbind(dim=1)
            out = 0.2999 * ch[0] + 0.587 * ch[1] + 0.114 * ch[2]
            out = out.unsqueeze(dim=0).expand(input.shape)
            return out.to(input.device, input.dtype)
        return input

class Grayscale(Transform):
    def forward(self, input: Tensor) -> Tensor:
        gray = RandomGrayscale(chance=(1, 1))
        return gray(input)

class ToSepia(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Equalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class AutoContrast(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Solarize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Posterize(Transform):
    def __init__(self) -> None:
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
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class HueSaturationValue(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class TonemapHDR(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

