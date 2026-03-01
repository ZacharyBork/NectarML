import colorsys
from PIL import Image, ImageEnhance
import numpy as np

import _nectarml
from nectarml.vision.transforms import Transform
from nectarml.benchmark import benchmark_time

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
    
    def forward(self, input: Image.Image) -> Image.Image:
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
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomContrast(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomSaturation(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass
    
class RandomHue(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomGamma(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class Grayscale(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomGrayscale(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class ToSepia(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class Equalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class AutoContrast(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class Solarize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class Posterize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class Invert(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class CLAHE(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class ChannelShuffle(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class ChannelDropout(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RGBShift(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class HueSaturationValue(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class TonemapHDR(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

