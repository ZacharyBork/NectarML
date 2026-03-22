import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

_rng = np.random.default_rng()

class GaussianNoise(Transform):
    def __init__(
        self,
        std_range: tuple[float, float] = (0.1, 0.2),
        mean_range: tuple[float, float] = (0.0, 0.0),
        per_channel: bool = False,
        noise_scale_factor: float = 1.0    
    ) -> None:
        super().__init__()
        self.std = std_range
        self.mean = mean_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        loc = _rng.random(()) * (self.mean[1] - self.mean[0]) + self.mean[0]
        scale = _rng.random(()) * (self.std[1] - self.std[0]) + self.std[0]
        loc *= max_value
        scale *= max_value
        
        if self.per_channel: noise_shape = input.shape
        else: noise_shape = (input.shape[0], 1) + input.shape[2:]

        rand = _rng.normal(loc, scale, noise_shape).astype(input.dtype)        
        rand *= self.noise_scale_factor
        noise = Tensor(rand, rand.shape).to(input.device, input.dtype)

        if not self.per_channel: noise = noise.expand(input.shape)
        output = (input + noise).clamp(0.0, max_value)
        return output

class SaltAndPepperNoise(Transform):
    def __init__(
        self,
        amount: tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper: tuple[float, float] = (0.4, 0.6)
    ) -> None:
        super().__init__()
        amt = _rng.random() * (amount[1] - amount[0]) + amount[0]
        self.salt_vs_pepper = (
            salt_vs_pepper[0] * amt,
            salt_vs_pepper[1] * amt)
    
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()

        shape = (input.shape[0], 1) + input.shape[2:]
        salt_arr = (_rng.random(size=shape) < self.salt_vs_pepper[0])
        salt_arr = salt_arr.astype(input.dtype)
        pepper_arr = (_rng.random(size=shape) < self.salt_vs_pepper[1])
        pepper_arr = (1 - pepper_arr).astype(input.dtype)
        
        salt = Tensor(salt_arr**10, shape) * max_value
        pepper = Tensor(pepper_arr**10, shape) * max_value
        
        salt = salt.expand(input.shape).to(input.device, input.dtype)
        pepper = pepper.expand(input.shape).to(input.device, input.dtype)
        
        output = (F.minimum(pepper, F.maximum(input, salt)))
        return output.clamp(0.0, max_value)

class SpeckleNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        _rng.gamma()

class ISONoise(Transform):
    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5)
    ) -> None:
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class MultiplicativeNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

