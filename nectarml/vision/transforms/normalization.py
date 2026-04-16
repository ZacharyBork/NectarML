from typing import Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.typing import float16, float32, uint8
from nectarml.vision.transforms.transform import Transform 
from nectarml.vision.transforms.common    import TransformInput

class Normalize(Transform):
    def __init__(
        self,
        mean:    list[float],
        std:     list[float],
        eps:     float = 1e-8,
        inplace: bool  = False
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
        self.inplace = inplace
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        assert input.ndim >= 3, \
            'Normalize expects at least 3D input [C, ...] or [B, C, ...]'
    
        if   input.ndim == 3: broadcast_shape = (len(self.mean), 1, 1)
        elif input.ndim == 4: broadcast_shape = (1, len(self.mean), 1, 1)
        elif input.ndim == 5: broadcast_shape = (1, len(self.mean), 1, 1, 1)
        
        mean = Tensor(np.array(self.mean, dtype=np.float32))
        mean = mean.reshape(broadcast_shape).to(input.device, input.dtype)
        std  = Tensor(np.array(self.std,  dtype=np.float32))
        std  = std.reshape(broadcast_shape).to(input.device, input.dtype)
        
        out = (input - mean) / (std + self.eps)
        if self.inplace: return input.copy_(out)
        return out
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Denormalize(Transform):
    def __init__(
        self,
        mean:    list[float],
        std:     list[float],
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        if   input.ndim == 3: broadcast_shape = (len(self.mean), 1, 1)
        elif input.ndim == 4: broadcast_shape = (1, len(self.mean), 1, 1)
        elif input.ndim == 5: broadcast_shape = (1, len(self.mean), 1, 1, 1)

        mean = Tensor(np.array(self.mean, dtype=np.float32))
        mean = mean.reshape(broadcast_shape).to(input.device)
        std  = Tensor(np.array(self.std, dtype=np.float32))
        std  = std.reshape(broadcast_shape).to(input.device)

        out = input * std + mean
        if self.inplace: return input.copy_(out)
        return out

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class MinMaxNormalize(Transform):
    def __init__(
        self,
        min_value: int | float = 0.0,
        max_value: int | float = 1.0,
        inplace:          bool = False
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.inplace = inplace
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _min, _max = input.min().item(), input.max().item()
        if _min == _max:
            return input * 0.0 + self.min_value + \
                (self.max_value - self.min_value) * 0.5
        _range = self.max_value - self.min_value
        out = (input - _min) * (_range / (_max - _min)) + self.min_value
        if self.inplace: return input.copy_(out)
        return out

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToFloat(Transform):
    def __init__(
        self, 
        half_precision: bool = False,
        scale:          bool = False
    ) -> None:
        super().__init__()
        self.output_dtype = float16 if half_precision else float32
        self.scale = scale
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        out = input.to(dtype=self.output_dtype)
        if self.scale and input.dtype == uint8: out = out / 255.0
        return out
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToUint8(Transform):
    def __init__(self, scale: bool = False) -> None:
        super().__init__()
        self.norm = MinMaxNormalize(0.0, 255.0) if scale else None
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        if self.norm is not None: input = self.norm(input)
        return input.round().to(dtype=uint8)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

