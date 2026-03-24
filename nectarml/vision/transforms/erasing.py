import numpy as np
from PIL import Image
from typing import Literal

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike
from nectarml.creation import zeros, rand, ones
from nectarml.vision.transforms import Transform

class RandomErasing(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class CoarseDropout(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

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
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomShadow(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

