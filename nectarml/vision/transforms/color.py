import colorsys
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

import _nectarml
import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import full
from nectarml.vision.transforms.transform import Transform, TransformInput 
from nectarml.cuda.utils import map_dtype
from nectarml.random import RNG

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

        return _hsv_adjust(input, self._hue, self._sat, self._val, max_value)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._hue      = self._random_in_range(self.hue)
        self._sat      = self._random_in_range(self.saturation)
        self._val      = self._random_in_range(self.brightness)
        self._contrast = self._random_in_range(self.contrast)

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
        super().__init__()
        self.value_range = value_range
        self.p = p
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return _hsv_adjust(input,  0.0, 1.0, self._brightness)
    
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance: return input
        self._brightness = self._random_in_range(self.value_range)
    
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
        super().__init__()
        self.value_range = value_range
        self.p = p
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        out = (((input / max_value) - 0.5) * self._contrast + 0.5) * max_value
        return out.clamp(0.0, max_value)
    
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance or np.allclose(list(self.value_range), [1, 1]): 
            return input
        
        self._contrast = self._random_in_range(self.value_range)
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
        super().__init__()
        self.value_range = value_range
        self.p = p
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return _hsv_adjust(input,  0.0, self._saturation, 1.0)
    
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance: return input
        self._saturation = self._random_in_range(self.value_range)
    
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
        super().__init__()
        self.value_range = value_range
        self.p = p
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return _hsv_adjust(input, self._hue, 1.0, 1.0)
    
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance: return input
        self._hue = self._random_in_range(self.value_range)
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
        super().__init__()
        self.value_range = value_range
        self.p = p
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        out = (input / max_value) ** self._gamma * max_value
        return out.clamp(0.0, max_value)
    
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance or np.allclose(list(self.value_range), [1, 1]): 
            return input
        
        self._gamma = self._random_in_range(self.value_range)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToGrayscale(Transform):
    def __init__(self) -> None:
        super().__init__()
        
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

class RandomGrayscale(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.to_grayscale = ToGrayscale()
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return self.to_grayscale(input)
    
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance: return input
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToSepia(Transform):
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
    
class RandomSepia(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.to_sepia = ToSepia()
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return self.to_sepia(input)
        
    def forward(self, input: TransformInput) -> TransformInput:
        chance = self._random_in_range()
        if self.p <= chance: return input
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
        by_channel: bool = True
    ) -> None:
        super().__init__()
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

class AutoContrast(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Solarize(Transform):
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
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        r, g, b = (input / max_value).unbind(dim=1)
        channels = [
            F.where((r < self._thresholds[0]), r, 1-r),
            F.where((g < self._thresholds[1]), g, 1-g),
            F.where((b < self._thresholds[2]), b, 1-b)]
        return (F.stack(channels, dim=1) * max_value).clamp(0.0, max_value)
    
    def forward(self, input: TransformInput) -> TransformInput:
        if self.per_channel:
            self._thresholds = [
                self._random_in_range(self.threshold_range)
                for _ in range(3)]
        else: self._thresholds = [
            self._random_in_range(self.threshold_range)] * 3
        
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class Posterize(Transform):
    def __init__(self, levels: int = 10) -> None:
        super().__init__()
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

class CLAHE(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ChannelShuffle(Transform):
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        ch = input.unbind(dim=1)
        shuffled = [ch[i] for i in self._channels]
        return F.stack(shuffled, dim=1).to(input.device, input.dtype)

    def forward(self, input: TransformInput) -> TransformInput:
        self._channels = list(range(input.image.shape[1]))
        RNG.shuffle(self._channels)
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
        fill: float = 0.0
    ) -> None:
        super().__init__()
        self.range = range
        self.fill = fill
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        channels = input.unbind(dim=1)
        
        new_channel = full(channels[0].shape, self.fill) * max_value
        channels[self._index] = new_channel.to(input.device, input.dtype)
        return F.stack(channels, dim=1).clamp(0.0, max_value)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._index = RNG.randint(self.range[0], self.range[1])
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
        b_shift_limit: tuple[int, int] = (-20, 20)
    ) -> None:
        super().__init__()
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
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._r = self._random_in_range(self.r_shift_limit)
        self._g = self._random_in_range(self.g_shift_limit)
        self._b = self._random_in_range(self.b_shift_limit)
        
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class HueSaturationValue(Transform[Tensor, Tensor]):
    def __init__(
        self,
        hue: float = 0.0,
        saturation: float = 1.0,
        value: float = 1.0
    ) -> None:
        super().__init__()
        self.hue = hue
        self.sat = saturation
        self.val = value
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return _hsv_adjust(input, self.hue, self.sat, self.val)
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class TonemapHDR(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

