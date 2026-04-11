import io
from PIL import Image
from typing import Literal

import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import Size, float32
from nectarml.vision.transforms.transform import Transform 
from nectarml.vision.transforms.common import TransformInput

class GaussianNoise(Transform):
    def __init__(
        self,
        std_range: tuple[float, float] = (0.1, 0.2),
        mean_range: tuple[float, float] = (0.0, 0.0),
        per_channel: bool = False,
        noise_scale_factor: float = 1.0,
        p: float = 0.5 
    ) -> None:
        super().__init__(p=p)
        self.std_range = std_range
        self.mean_range = mean_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        self._rand = self._rand.astype(input.dtype) * self.noise_scale_factor
        noise = Tensor(self._rand, self._rand.shape, input.dtype, input.device)

        if not self.per_channel: noise = noise.expand(input.shape)
        return ((norm + noise) * max_value).clamp(0.0, max_value)

    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        loc = self._random_in_range(self.mean_range)
        scale = self._random_in_range(self.std_range)
        
        if self.per_channel: noise_shape = input_shape
        else: noise_shape = (input_shape[0], 1) + input_shape[2:]
        self._rand = self.rng.normal(loc, scale, noise_shape)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class SaltAndPepperNoise(Transform):
    def __init__(
        self,
        amount: tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper: tuple[float, float] = (0.4, 0.6),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        salt = self._salt.expand(input.shape).to(input.device, input.dtype)
        pepper = self._pepper.expand(input.shape).to(input.device, input.dtype)
        
        output = (F.minimum(pepper, F.maximum(norm, salt))) * max_value
        return output.clamp(0.0, max_value)
    
    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        amt = self._random_in_range(self.amount)

        shape = (input_shape[0], 1) + input_shape[2:]
        salt_arr = (self.rng.random(size=shape) < self.salt_vs_pepper[0]*amt)
        salt_arr = salt_arr.astype(float32)
        pepper_arr = (self.rng.random(size=shape) < self.salt_vs_pepper[1]*amt)
        pepper_arr = (1 - pepper_arr).astype(float32)
        
        self._salt = Tensor(salt_arr**10, shape, dtype=float32)
        self._pepper = Tensor(pepper_arr**10, shape, dtype=float32)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class SpeckleNoise(Transform):
    def __init__(
        self,
        std_range: tuple[float, float] = (0.1, 0.2),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.std_range = std_range
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        self._noise = self._noise.to(input.device, input.dtype)
        out = input + input * self._noise
        return out.clamp(0.0, input.max().item())
    
    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        std = self._random_in_range(self.std_range)
        arr = self.rng.normal(0, std, input_shape).astype(float32)
        self._noise = Tensor(arr, dtype=float32)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ISONoise(Transform):
    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.color_shift = color_shift
        self.intensity = intensity
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        r, g, b = norm.unbind(dim=1)
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
        luma_weight = (1.0 - luma).expand(input.shape)

        self._luma_noise = self._luma_noise.to(input.device, input.dtype)
        self._color_noise = self._color_noise.to(input.device, input.dtype)
        out = norm + self._luma_noise * luma_weight + self._color_noise
        return (out * max_value).clamp(0.0, max_value)
        
    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        luma_std = self._random_in_range(self.intensity)
        _luma = self.rng.normal(0, luma_std, input_shape).astype(float32)
        self._luma_noise = Tensor(_luma, dtype=float32)
        
        color_std = self._random_in_range(self.color_shift)
        _color = self.rng.normal(0, color_std, input_shape).astype(float32)
        self._color_noise = Tensor(_color, dtype=float32)
        
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class MultiplicativeNoise(Transform):
    def __init__(
        self,
        multiplier_range: tuple[float, float] = (0.8, 1.2),
        per_channel: bool = False,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        self._noise = self._noise.to(input.device, input.dtype)
        return (input * self._noise).clamp(0.0, input.max().item())
        
    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        if self.per_channel:
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else: noise_shape = input_shape
        
        r = self.multiplier_range
        arr = self.rng.uniform(r[0], [1], noise_shape).astype(float32)
        self._noise = Tensor(arr, dtype=float32)
        
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class ImageCompression(Transform):
    def __init__(
        self,
        compression_type: Literal['jpeg', 'webp'] = 'jpeg',
        quality_range: int | tuple[int, int] = (50, 95),
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.compression_type = compression_type
        self.quality = (quality_range, quality_range) \
            if isinstance(quality_range, int) else quality_range

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        arr = input.cpu().numpy()[0] / input.max().item()
        arr = (arr * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(arr, mode='RGB')

        buf = io.BytesIO()
        f = self.compression_type.upper()
        img.save(buf, format=f, quality=self._quality)
        buf.seek(0)
        compressed = Image.open(buf).convert('RGB')

        result = np.array(compressed).astype(np.float32) / 255.0
        result = result.transpose(2, 0, 1)[np.newaxis]

        return Tensor(result, dtype=input.dtype, device=input.device)
            
    def _build_parameters(self) -> None:
        self._quality = int(self._random_in_range(self.quality))

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints)

