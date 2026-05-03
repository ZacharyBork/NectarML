from warnings import warn
from typing import Any

import numpy as np
from PIL import Image

from nectarml.core   import Tensor
from nectarml.typing import DeviceLikeType, dtype, float32
from nectarml.vision.transforms.transform import Transform, UtilityTransform
from nectarml.vision.transforms.common    import TransformInput

class ToTensor(Transform):
    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        
    def _transform(
        self, 
        input: np.ndarray | Image.Image | Tensor | None
    ) -> Tensor:
        if input is None or isinstance(input, Tensor): return input
        
        if isinstance(input, Image.Image):
            data = np.array(input)
            data = data.transpose((2, 0, 1))[np.newaxis]
        elif isinstance(input, np.ndarray): 
            data = input
            if data.ndim == 2: data = data[np.newaxis]
            if data.ndim == 3: data = data[np.newaxis]
        else: raise ValueError(f'Unsupported input type: {type(input)}')
        
        output = Tensor(data.astype(np.float32), dtype=float32, device='cpu')
        if self.normalize: output = output / max(output.max().item(), 1.0)
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
        normalize:   bool = True,
        value_range: tuple[int, int] = (0, 255)
    ) -> None:
        super().__init__()
        self.normalize   = normalize
        self.value_range = value_range

    def _remap(self, input: np.ndarray) -> np.ndarray:
        vmin, vmax = input.min(), input.max()
        rmin, rmax = self.value_range
        if vmin == vmax: return input * 0.0 + rmin + (rmax - rmin) * 0.5
        return (input - vmin) * ((rmax - rmin) / (vmax - vmin)) + rmin
    
    def forward(self, input: Tensor | np.ndarray) -> Image.Image:
        if   isinstance(input, Image.Image): return input
        elif isinstance(input, np.ndarray):  pass
        elif isinstance(input, Tensor): input = input.cpu().numpy()
        else: raise ValueError(
            f'ToPIL recieved invalid input type: {type(input)}')
        
        assert input.ndim == 4, \
            'ToPIL expects input to have ndim=4 (B, C, H, W).'
        assert input.shape[0] == 1, \
            'ToPIL expects input to have only 1 batch ([1, C, H, W])'

        output = np.ascontiguousarray(input.squeeze(axis=0))
        output = output.transpose((1, 2, 0))
        if self.normalize: output = self._remap(output)
        
        if np.any(~np.isfinite(output)):
            warn('ToPIL encountered non-finite value during image conversion.')
            output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
        return Image.fromarray(output.astype(np.uint8), 'RGB')

class ToNumpy(UtilityTransform[Tensor | Image.Image, np.ndarray]):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor | Image.Image) -> np.ndarray:
        if isinstance(input, np.ndarray): return input
        
        if   isinstance(input, Tensor):      return input.numpy()
        elif isinstance(input, Image.Image): return np.array(input)
        else: raise ValueError(f'Unsupported input type: {type(input)}')
        
class FromTorch(Transform):
    def __init__(self) -> None:
        super().__init__(p=1.0)
        
    def _transform(self, input: Any) -> Tensor:
        if input is None: return input
        from   nectarml.compat import pytorch
        return pytorch.tensor_torch2nectar(input)
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = self._transform(input.boxes),
            keypoints = self._transform(input.keypoints)
        ) 
        
class ToTorch(Transform):
    def __init__(self) -> None:
        super().__init__(p=1.0)
        
    def _transform(self, input: Tensor) -> Any:
        if input is None: return input
        from   nectarml.compat import pytorch
        return pytorch.tensor_nectar2torch(input)
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = self._transform(input.boxes),
            keypoints = self._transform(input.keypoints)
        ) 

class ConvertDtype(Transform):
    def __init__(
        self,
        new_dtype:     dtype = float32,
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
        new_device: DeviceLikeType = 'cpu'
    ) -> None:
        super().__init__()
        self.new_device = new_device
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input  
        return input.to(self.new_device)  
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = self._transform(input.boxes),
            keypoints = self._transform(input.keypoints)
        )
    
class ToCPU(Transform):
    def __init__(self) -> None:
        super().__init__()
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.contiguous().cpu()
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = self._transform(input.boxes),
            keypoints = self._transform(input.keypoints)
        )
    
class ToCUDA(Transform):
    def __init__(self) -> None:
        super().__init__()

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.contiguous().cuda()

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = self._transform(input.boxes),
            keypoints = self._transform(input.keypoints)
        )
    
class Cast(Transform):
    def __init__(
        self,
        new_device:     DeviceLikeType | None = None,
        new_dtype:      dtype | None = None, 
        transform_mask: bool = False
    ) -> None:
        super().__init__()
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
    def __init__(self) -> None:
        super().__init__()
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return input.contiguous()
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask),
            boxes     = self._transform(input.boxes),
            keypoints = self._transform(input.keypoints)
        )

