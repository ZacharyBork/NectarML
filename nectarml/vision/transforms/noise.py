import numpy as np

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
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class SpeckleNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ISONoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class MultiplicativeNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

