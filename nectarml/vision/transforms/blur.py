from typing import Literal

import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import float32
from nectarml.vision.transforms.transform import Transform, TransformInput

class GaussianBlur(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

    def forward(self, input: TransformInput) -> TransformInput:
        pass

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
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

    def forward(self, input: TransformInput) -> TransformInput:
        pass

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
