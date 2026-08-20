import numpy as np

import nectarml.nn.functional as F
from nectarml.core        import Tensor
from nectarml.typing      import float16, float32, uint8
from nectarml.vision.transforms.transform import Transform 
from nectarml.vision.transforms.common    import TransformInput

class Normalize(Transform):
    def __init__(
        self,
        mean:    float | list[float],
        std:     float | list[float],
        eps:     float = 1e-8,
        inplace: bool  = False
    ) -> None:
        '''Normalizes inputs to a given mean and standard deviation.

        Args:
            mean    : The mean value for the normalized output's range.
            std     : The standard deviation for the normalized output's range.
            eps     : Small epsilon value to avoid zero-division error.
            inplace : If True, this tranform will modify input tensors in-
                      place, rather than creating new tensors for output. This
                      can help save a small amount of memory overhead.
        '''
        super().__init__()
        self.mean    = mean
        self.std     = std
        self.eps     = eps
        self.inplace = inplace
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        device, dtype = input.device, input.dtype
        
        _mean, _std = self.mean, self.std
        if isinstance(_mean, int | float): _mean = [_mean] * input.shape[1]
        if isinstance(_std,  int | float): _std  = [_std]  * input.shape[1]
                        
        mean = Tensor(np.array(_mean, dtype=np.float32))
        mean = mean.reshape((1, len(_mean), 1, 1)).to(device, dtype)
        
        std = Tensor(np.array(_std,  dtype=np.float32))
        std = std.reshape((1, len(_std), 1, 1)).to(device, dtype)
        
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
        mean:    float | list[float],
        std:     float | list[float],
        inplace: bool = False
    ) -> None:
        '''Denormalizes inputs from a given mean and standard deviation.

        Args:
            mean    : The mean value of the range to denormalize from.
            std     : The standard deviation of the range to denormalize from.
            inplace : If True, this tranform will modify input tensors in-
                      place, rather than creating new tensors for output. This
                      can help save a small amount of memory overhead.
        '''
        super().__init__()
        self.mean    = mean
        self.std     = std
        self.inplace = inplace

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        device, dtype = input.device, input.dtype
        
        _mean, _std = self.mean, self.std
        if isinstance(_mean, int | float): _mean = [_mean] * input.shape[1]
        if isinstance(_std,  int | float): _std  = [_std]  * input.shape[1]
            
        mean = Tensor(np.array(_mean, dtype=np.float32))
        mean = mean.reshape((1, len(_mean), 1, 1)).to(device, dtype)
        
        std = Tensor(np.array(_std, dtype=np.float32))
        std = std.reshape((1, len(_std), 1, 1)).to(device, dtype)
        
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
        per_batch:        bool = False,
        inplace:          bool = False
    ) -> None:
        '''Normalizes inputs to a defined min and max value.
        
        Args:
            min_value : The minimum value for the output's range.
            max_value : The minimum value for the output's range.
            per_batch : If True, each item in a batch of the input tensors will
                        be  normalized separately, ensuring each item fully
                        saturates the range. Otherwise, batched tensors will be
                        normalized as a whole, so that the full batch saturates
                        the desired range.
            inplace   : If True, this tranform will modify input tensors in-
                        place, rather than creating new tensors for output. 
                        This can help save a small amount of memory overhead.
        '''
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.per_batch = per_batch
        self.inplace   = inplace
        
    def _remap(self, input: Tensor) -> Tensor:
        _min, _max = input.min().item(), input.max().item()
        if _min == _max:
            return input * 0.0 + self.min_value \
                 + (self.max_value - self.min_value) * 0.5
        _range = self.max_value - self.min_value
        return (input - _min) * (_range / (_max - _min)) + self.min_value
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        if self.per_batch and input.shape[0] > 1:
            batches = input.unbind(dim=0, keepdim=True)
            norm    = [self._remap(i) for i in batches]
            out     = F.cat(norm, dim=0)
        else: out = self._remap(input)
        
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
        '''Casts input tensors to float.
        
        Args:
            half_precision : If True, the input tensors will be cast to 
                             float16, otherwise they will be cast to float32.
            scale          : If True and the input tensor's DType is uint8 
                             (standard for images loaded with PIL), the output
                             tensor's range will be normalized (0-1). If False,
                             the value range will remain as is after the cast.
        '''
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
        '''Casts input tensors to uint8.
        
        Args:
            scale : If True, input tensors will have their value ranges
                    automatically normalized to saturate the full (0-255) range
                    of uint8 before being cast. Otherwise the tensors will be
                    cast with their original value ranges.
        '''
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

