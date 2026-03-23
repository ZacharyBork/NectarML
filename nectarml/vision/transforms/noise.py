import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

class GaussianNoise(Transform[Tensor, Tensor]):
    def __init__(
        self,
        std_range: tuple[float, float] = (0.1, 0.2),
        mean_range: tuple[float, float] = (0.0, 0.0),
        per_channel: bool = False,
        noise_scale_factor: float = 1.0    
    ) -> None:
        super().__init__()
        self.std_range = std_range
        self.mean_range = mean_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        norm = input / max_value
        
        loc = self._random_in_range(self.mean_range)
        scale = self._random_in_range(self.std_range)
        
        if self.per_channel: noise_shape = input.shape
        else: noise_shape = (input.shape[0], 1) + input.shape[2:]

        rand = self.rng.normal(loc, scale, noise_shape).astype(input.dtype)        
        rand *= self.noise_scale_factor
        noise = Tensor(rand, rand.shape).to(input.device, input.dtype)

        if not self.per_channel: noise = noise.expand(input.shape)
        return ((norm + noise) * max_value).clamp(0.0, max_value)

class SaltAndPepperNoise(Transform[Tensor, Tensor]):
    def __init__(
        self,
        amount: tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper: tuple[float, float] = (0.4, 0.6)
    ) -> None:
        super().__init__()
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        norm = input / max_value
        
        amt = self._random_in_range(self.amount)

        shape = (input.shape[0], 1) + input.shape[2:]
        salt_arr = (self.rng.random(size=shape) < self.salt_vs_pepper[0]*amt)
        salt_arr = salt_arr.astype(input.dtype)
        pepper_arr = (self.rng.random(size=shape) < self.salt_vs_pepper[1]*amt)
        pepper_arr = (1 - pepper_arr).astype(input.dtype)
        
        salt = Tensor(salt_arr**10, shape)
        pepper = Tensor(pepper_arr**10, shape)
        
        salt = salt.expand(input.shape).to(input.device, input.dtype)
        pepper = pepper.expand(input.shape).to(input.device, input.dtype)
        
        output = (F.minimum(pepper, F.maximum(norm, salt))) * max_value
        return output.clamp(0.0, max_value)

class SpeckleNoise(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ISONoise(Transform[Tensor, Tensor]):
    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5)
    ) -> None:
        raise NotImplementedError
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class MultiplicativeNoise(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

