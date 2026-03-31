import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.vision.transforms.transform import Transform, TransformInput 

class GaussianNoise(Transform):
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
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        self._rand = self._rand.astype(input.dtype) * self.noise_scale_factor
        noise = Tensor(self._rand, self._rand.shape)
        noise = noise.to(input.device, input.dtype)

        if not self.per_channel: noise = noise.expand(input.shape)
        return ((norm + noise) * max_value).clamp(0.0, max_value)

    def forward(self, input: TransformInput) -> TransformInput:
        loc = self._random_in_range(self.mean_range)
        scale = self._random_in_range(self.std_range)
        
        if self.per_channel: noise_shape = input.image.shape
        else: noise_shape = (input.image.shape[0], 1) + input.image.shape[2:]
        self._rand = self.rng.normal(loc, scale, noise_shape)
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
        salt_vs_pepper: tuple[float, float] = (0.4, 0.6)
    ) -> None:
        super().__init__()
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
    
    def forward(self, input: TransformInput) -> TransformInput:
        amt = self._random_in_range(self.amount)

        shape = (input.image.shape[0], 1) + input.image.shape[2:]
        salt_arr = (self.rng.random(size=shape) < self.salt_vs_pepper[0]*amt)
        salt_arr = salt_arr.astype(input.image.dtype)
        pepper_arr = (self.rng.random(size=shape) < self.salt_vs_pepper[1]*amt)
        pepper_arr = (1 - pepper_arr).astype(input.image.dtype)
        
        self._salt = Tensor(salt_arr**10, shape)
        self._pepper = Tensor(pepper_arr**10, shape)
        
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
        std_range: tuple[float, float] = (0.1, 0.2)
    ) -> None:
        super().__init__()
        self.std_range = std_range
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        out = input + input * self._noise
        return out.clamp(0.0, input.max().item())
    
    def forward(self, input: TransformInput) -> TransformInput:
        dtype = input.image.dtype
        device = input.image.device
        
        std = self._random_in_range(self.std_range)
        arr = self.rng.normal(0, std, input.image.shape).astype(dtype)
        self._noise = Tensor(arr).to(device, dtype)
        
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
        intensity: tuple[float, float] = (0.1, 0.5)
    ) -> None:
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        r, g, b = norm.unbind(dim=1)
        luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
        luma_weight = (1.0 - luma).expand(input.shape)

        out = norm + self._luma_noise * luma_weight + self._color_noise
        return (out * max_value).clamp(0.0, max_value)
        
    def forward(self, input: TransformInput) -> TransformInput:
        dtype = input.image.dtype
        device = input.image.device
        
        luma_std = self._random_in_range(self.intensity)
        self._luma_noise = Tensor(
            self.rng.normal(0, luma_std, input.image.shape).astype(dtype)
        ).to(device, dtype)
        
        color_std = self._random_in_range(self.color_shift)
        self._color_noise = Tensor(
            self.rng.normal(0, color_std, input.image.shape).astype(dtype)
        ).to(device, dtype)
        
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
        per_channel: bool = False
    ) -> None:
        super().__init__()
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return (input * self._noise).clamp(0.0, input.max().item())
        
    def forward(self, input: TransformInput) -> TransformInput:
        dtype = input.image.dtype
        device = input.image.device
        
        if self.per_channel:
            noise_shape = (input.image.shape[0], input.image.shape[1], 1, 1)
        else: noise_shape = input.image.shape
        
        r = self.multiplier_range
        arr = self.rng.uniform(r[0], [1], noise_shape).astype(dtype)
        self._noise = Tensor(arr).to(device, dtype)
        
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
        

