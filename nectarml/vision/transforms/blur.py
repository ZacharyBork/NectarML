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
        kernel_size: int = 7,
        sigma:     float = 1.0,
        iterations:  int = 1
    ) -> None:
        super().__init__()
        assert kernel_size % 2 != 0, \
            '"kernel_size" must be an odd integer value.'
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.iterations = iterations
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        ks = self.kernel_size  
        center = ks // 2
            
        xx = linspace(0, ks-1, ks, dtype=float32, device=input.device) - center
        yy = linspace(0, ks-1, ks, dtype=float32, device=input.device) - center
        
        xx = xx.reshape((ks, 1)).expand((ks, ks))
        yy = yy.reshape((1, ks)).expand((ks, ks))
        
        kernel = (-(xx**2 + yy**2) / (2 * self.sigma**2)).exp()
        kernel = kernel / kernel.sum()
    
        output = input.clone()
        for _ in range(self.iterations):
            output = _apply_kernel_2d(output, kernel)
        
        return output

    def forward(self, input: TransformInput) -> TransformInput:
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
        kernel_size: int = 5,
        iterations:  int = 1
    ) -> None:
        super().__init__()
        assert kernel_size % 2 != 0, \
            '"kernel_size" must be an odd integer value.'
        self.kernel_size = kernel_size
        self.iterations = iterations
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        ks = self.kernel_size
        kernel = ones((ks, ks), dtype=float32, device=input.device) / (ks * ks)
        
        output = input.clone()
        for _ in range(self.iterations):
            output = _apply_kernel_2d(output, kernel)
        
        return output
    
    def forward(self, input: TransformInput) -> TransformInput:
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
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

    def forward(self, input: TransformInput) -> TransformInput:
        pass

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
