from typing import Literal

import numpy as np
from PIL import Image

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32, uint8
from nectarml.vision.transforms.transform import \
    Transform, UtilityTransform, TransformInput
from nectarml.vision.transforms.normalization import MinMaxNormalize

class ToTensor(Transform):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        
    def _transform(
        self, 
        input: np.ndarray | Image.Image | Tensor | None
    ) -> Tensor:
        if input is None or isinstance(input, Tensor): return input
        
        if isinstance(input, Image.Image):
            data = np.array(input).astype(float32)
        elif isinstance(input, np.ndarray): data = input
        else: raise ValueError(f'Unsupported input type: {type(input)}')
        
        output = Tensor(data, dtype=float32, device='cpu')
        output = output.permute((2, 0, 1)).unsqueeze(dim=0)
        if self.normalize: output = output / np.maximum(data.max(), 1.0)
        return output
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class ToPIL(UtilityTransform[Tensor | np.ndarray, Image.Image]):
    def __init__(
        self,
        normalize: bool = True,
        value_range: tuple[int, int] = (0, 255)
    ) -> None:
        super().__init__()
        if not normalize: self.norm = None
        else: self.norm = MinMaxNormalize(value_range[0], value_range[1])
    
    def forward(self, input: Tensor | np.ndarray) -> Image.Image:
        if isinstance(input, Image.Image): return input
        
        assert input.ndim in [3, 4], \
            'ToPIL expects input to be 3D ([C, H, W]) or 4D ([B, C, H, W])'

        if isinstance(input, np.ndarray):
            output = Tensor(input, input.shape, input.dtype)
        elif isinstance(input, Tensor): output = input.clone()
        else: raise ValueError(f'Unsupported input type: {type(input)}')
        
        if input.ndim == 4:
            assert input.shape[0] == 1, \
                'ToPIL expects input to have only 1 batch ([1, H, W])'
            output = output.squeeze(0) 
        
        output = output.permute((1, 2, 0)).contiguous()
        if self.norm is not None: output = self.norm(output)
        
        arr = output.numpy()
        if np.any(~np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        return Image.fromarray(arr.astype(uint8), 'RGB')

class ToNumpy(UtilityTransform[Tensor | Image.Image, np.ndarray]):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor | Image.Image) -> np.ndarray:
        if isinstance(input, np.ndarray): return input
        
        if isinstance(input, Tensor): return input.numpy()
        elif isinstance(input, Image.Image): return np.array(input)
        else: raise ValueError(f'Unsupported input type: {type(input)}')

class ConvertDtype(Transform):
    def __init__(
        self,
        new_dtype: DTypeLike = float32,
        transform_mask: bool = False
    ) -> None:
        super().__init__()
        self.new_dtype = new_dtype
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.to(dtype=self.new_dtype)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ChangeDevice(Transform):
    def __init__(
        self,
        new_device: Literal['cpu', 'cuda'] = 'cpu',
        transform_mask: bool = False
    ) -> None:
        super().__init__(new_device)
        self.new_device = new_device
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input  
        return input.to(self.new_device)  
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class ToCPU(Transform):
    def __init__(self, transform_mask: bool = False) -> None:
        super().__init__()
        self.transform_mask = transform_mask
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.contiguous().cpu()
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class ToCUDA(Transform):
    def __init__(self, transform_mask: bool = False) -> None:
        super().__init__()
        self.transform_mask = transform_mask

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.contiguous().cuda()

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class Cast(Transform):
    def __init__(
        self,
        new_device: Literal['cpu', 'cuda'] | None = None,
        new_dtype: DTypeLike | None = None, 
        transform_mask: bool = False
    ) -> None:
        super().__init__(new_device)
        self.new_device = new_device
        self.new_dtype = new_dtype
        self.transform_mask = transform_mask
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        dtype  = input.dtype  if self.new_dtype  is None else self.new_dtype
        device = input.device if self.new_device is None else self.new_device
        return input.to(device, dtype)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class ToContiguous(Transform):
    def __init__(self, transform_mask: bool = False) -> None:
        super().__init__()
        self.transform_mask = transform_mask
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.contiguous()
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

