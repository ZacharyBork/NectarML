import math
import warnings
from typing import Literal

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike
from nectarml.creation import zeros, rand, ones
from nectarml.vision.transforms import Transform

class Erasing(Transform[Tensor, Tensor]):
    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        fill: float = 0.0
    ) -> None:
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.fill = fill
    
    def _build_mask(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        mask = ones((B, 1, H, W), input.dtype, input.device)
        image_area = H * W
        
        for b in range(B):
            area = self._random_in_range(self.scale) * image_area
            aspect_ratio = self._random_in_range(self.ratio)

            hole_h = min(int(math.sqrt(area / aspect_ratio)), H)
            hole_w = min(int(math.sqrt(area * aspect_ratio)), W)
            
            cy = self.rng.integers(0, H)
            cx = self.rng.integers(0, W)
            
            pY = (max(0, cy - hole_h // 2), min(H, cy + hole_h // 2))
            pX = (max(0, cx - hole_w // 2), min(W, cx + hole_w // 2))
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mask[b, 0, pY[0]:pY[1], pX[0]:pX[1]] = 0.0
        
        return mask
    
    def forward(self, input: Tensor) -> Tensor:
        mask = self._build_mask(input)
        return input * mask + self.fill * input.max().item() * (1 - mask)

class CoarseDropout(Transform[Tensor, Tensor]):
    def __init__(
        self,
        num_holes_range: tuple[int, int] = (1, 2),
        holes_height_range: tuple[float, float] = (0.1, 0.2),
        holes_width_range: tuple[float, float] = (0.1, 0.2),
        fill: float = 0.0
    ) -> None:
        super().__init__()
        self.num_holes_range = num_holes_range
        self.holes_height_range = holes_height_range
        self.holes_width_range = holes_width_range
        self.fill = fill
    
    def _build_mask(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        mask = ones((B, 1, H, W), input.dtype, input.device)
        
        for b in range(B):
            num_holes = int(round(self._random_in_range(self.num_holes_range)))
            for _ in range(num_holes):
                hole_h = int(self._random_in_range(self.holes_height_range)*H)
                hole_w = int(self._random_in_range(self.holes_width_range)*W)
                
                cy = self.rng.integers(0, H)
                cx = self.rng.integers(0, W)
                
                pY = (max(0, cy - hole_h // 2), min(H, cy + hole_h // 2))
                pX = (max(0, cx - hole_w // 2), min(W, cx + hole_w // 2))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mask[b, 0, pY[0]:pY[1], pX[0]:pX[1]] = 0.0
        
        return mask
    
    def forward(self, input: Tensor) -> Tensor:
        mask = self._build_mask(input)
        return input * mask + self.fill * input.max().item() * (1 - mask)

class GridDropout(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomSunFlare(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomFog(Transform[Tensor, Tensor]):
    def __init__(
        self,
        scale: int = 100,
        octaves: int = 4,
        intensity_range: tuple[float, float] = (0.3, 0.7),
        fog_color: tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        super().__init__()
        self.scale = scale
        self.octaves = octaves
        self.intensity_range = intensity_range
        self.fog_color = fog_color
        
    def _perlin_approx(
        self, 
        H: int, 
        W: int,
        device: Literal['cpu', 'cuda'],
        dtype: DTypeLike
    ) -> Tensor:
        noise = zeros((1, 1, H, W), dtype, device)
        amplitude = frequency = 1.0
        
        for _ in range(self.octaves):
            octave_h = max(1, int(H * frequency / self.scale))
            octave_w = max(1, int(W * frequency / self.scale))
            octave = F.upsample(
                rand((1, 1, octave_h, octave_w), dtype, device),
                size=(H, W), mode='bilinear')
            
            noise = noise + amplitude * octave
            amplitude *= 0.5
            frequency *= 2.0
        
        min_value = noise.min().item()
        return (noise - min_value) / (noise.max().item() - min_value)

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        max_value = input.max().item()
        intensity = self._random_in_range(self.intensity_range)
        
        fog_maps = F.stack(
            [self._perlin_approx(H, W, input.device, input.dtype).squeeze(0)
            for _ in range(B)], dim=0)
        
        fog_color = Tensor(self.fog_color, (3,), input.dtype, input.device)
        fog_color = fog_color.reshape((1, 3, 1, 1))

        fog = (1 - fog_maps * intensity) + fog_color * fog_maps * intensity
        
        out = input + fog
        return (out / out.max().item() * max_value).clamp(0.0, max_value)
        
class RandomRain(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomSnow(Transform[Tensor, Tensor]):
    def __init__(
        self,
        brighness_coef: float = 1.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3)
    ) -> None:
        super().__init__()
        self.brightness_coef = brighness_coef
        self.snow_point_range = snow_point_range
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        norm = input / max_value
        
        r, g, b = norm.unbind(dim=1)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
        
        snow_point = self._random_in_range(self.snow_point_range)
        snow_mask = ((luminance-snow_point) / (1.0-snow_point)).clamp(0.0, 1.0)
        
        noise = rand(snow_mask.shape, input.dtype, input.device)
        snow_mask = (snow_mask + noise * 0.1).clamp(0.0, 1.0)
        
        out = norm + snow_mask * (1.0 - norm) * self.brightness_coef
        return (out * max_value).clamp(0.0, max_value)

class RandomShadow(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

