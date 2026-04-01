from typing import Literal

import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import linspace, ones
from nectarml.typing import float32
from nectarml.vision.transforms.transform import Transform, TransformInput

### UTILS ###

def _apply_kernel_2d(image: Tensor, kernel: Tensor) -> Tensor:
    B, C, H, W = image.shape
    KH, KW = kernel.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    image_flat = image.reshape((B * C, 1, H, W))
    result = F.conv2d(image_flat, kernel, padding=(KH//2, KW//2), groups=1)
    return result.reshape((B, C, H, W))

### TRANSFORMS ###

class GaussianBlur(Transform):
    def __init__(
        self,
        kernel_size: int | tuple[int, int] = (3, 7),
        sigma: float | tuple[float, float] = 1.0,
        iterations:  int | tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.sigma = (sigma, sigma) \
            if isinstance(sigma, float | int) else sigma
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        ks = self._ks
        
        x = linspace(0, ks-1, ks, dtype=float32, device=input.device) - (ks//2)
        xx = x.reshape((ks, 1)).expand((ks, ks))
        yy = x.clone().reshape((1, ks)).expand((ks, ks))
        
        kernel = (-(xx**2 + yy**2) / (2 * self._sigma**2)).exp()
        kernel = kernel / kernel.sum()
    
        output = input.clone()
        for _ in range(self._iters): output = _apply_kernel_2d(output, kernel)
        return output

    def forward(self, input: TransformInput) -> TransformInput:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        self._ks = int(self.rng.choice(valid_sizes))
        self._sigma = self._random_in_range(self.sigma)
        self._iters = int(self._random_in_range(self.iterations))
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class MotionBlur(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
    
    def forward(self, input: TransformInput) -> TransformInput:
        pass

class MedianBlur(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

    def forward(self, input: TransformInput) -> TransformInput:
        pass

class BoxBlur(Transform):
    def __init__(
        self,
        kernel_size: int | tuple[int, int] = (3, 7),
        iterations:  int | tuple[int, int] = 1
    ) -> None:
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        kernel = (
            ones((self._ks, self._ks), device=input.device)
          / (self._ks * self._ks))
        
        output = input.clone()
        for _ in range(self._iters): output = _apply_kernel_2d(output, kernel)
        return output
    
    def forward(self, input: TransformInput) -> TransformInput:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        self._ks = int(self.rng.choice(valid_sizes))
        self._iters = int(self._random_in_range(self.iterations))
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomBlur(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

    def forward(self, input: TransformInput) -> TransformInput:
        pass

class Sharpen(Transform):
    def __init__(
        self,
        alpha: float | tuple[float, float] = 1.0,
        iterations:  int | tuple[int, int] = 1,
        kernel_size:     int = 3,
        sigma:         float = 1.0,
        blur_iterations: int = 1,
        method: Literal['gaussian', 'box'] = 'gaussian'
    ) -> None:
        super().__init__()
        self.alpha = (alpha, alpha) \
            if isinstance(kernel_size, int | float) else alpha
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
            
        match method:
            case 'box': 
                blur_fn = BoxBlur(kernel_size, blur_iterations)
            case 'gaussian': 
                blur_fn =  GaussianBlur(kernel_size, sigma, blur_iterations)
            case _: raise ValueError(f'Invalid blur method: {method}')
        self.blur = lambda x : blur_fn._transform(x)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        output = input.clone()
        for _ in range(self._iters):
            output = output + self._alpha * (output - self.blur(output))
            output = output.clamp(0.0, max_value)
        return output

    def forward(self, input: TransformInput) -> TransformInput:
        self._alpha = self._random_in_range(self.alpha)
        self._iters = int(self._random_in_range(self.iterations))
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Emboss(Transform):
    def __init__(
        self,
        kernel_mode: int = 0,
        rotate_kernel: bool = False,
        gray_level: int = 75
    ) -> None:
        super().__init__()
        self.gray_level = gray_level / 255
        
        match kernel_mode:
            case 0: 
                k = [[0,  1, 0],
                     [0,  0, 0],
                     [0, -1, 0]]
            case 1: 
                k = [[1, 0,  0],
                     [0, 0,  0],
                     [0, 0, -1]]
            case 2: 
                k = [[0, 0,  0],
                     [1, 0, -1],
                     [0, 0,  0]]
            case 3: 
                k = [[ 0, 0, 1],
                     [ 0, 0, 0],
                     [-1, 0, 0]]
            case _: 
                raise ValueError(
                    f'Kernel type expected values between 0-3 but found '
                    f'value: {kernel_mode}')
        
        k = np.array([[k]]).astype(float32)
        if rotate_kernel: k = np.rot90(k)
        self.kernel = Tensor(k, dtype=float32)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        kernel = self.kernel.to(input.device, input.dtype)

        gray = norm.mean(dim=1, keepdim=True)
                
        out = F.conv2d(gray, kernel, padding=1)
        out = F.sqrt(out ** 2 + 1e-6)
        out = F.maximum(out, self.gray_level)
        outputs = [out]*3

        return (F.cat(outputs, dim=1) * max_value).clamp(0.0, max_value)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class UnsharpMask(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

    def forward(self, input: TransformInput) -> TransformInput:
        pass
