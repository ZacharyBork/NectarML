from typing import Literal

import numpy as np
from PIL import Image

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.vision.transforms import Transform
from nectarml.vision.transforms.normalization import MinMaxNormalize

class ToTensor(Transform[np.ndarray | Image.Image, Tensor]):
    def __init__(
        self, 
        device: Literal['auto', 'cpu', 'cuda'] = 'auto',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device)
        self.dtype = dtype
    
    def forward(self, input: np.ndarray | Image.Image) -> Tensor:
        if isinstance(input, Image.Image):
            data = np.array(input).astype(self.dtype)
        elif isinstance(input, np.ndarray): data = input
        else: raise ValueError(f'Unsupported input type: {type(input)}')
        
        output = Tensor(data, dtype=self.dtype, device=self.device)
        return output.permute((2, 0, 1)).unsqueeze(dim=0)

class ToPIL(Transform[Tensor | np.ndarray, Image.Image]):
    def __init__(
        self,
        normalize: bool = False,
        value_range: tuple[int, int] = (0, 255)
    ) -> None:
        super().__init__()
        if not normalize: self.norm = None
        else: self.norm = MinMaxNormalize(value_range[0], value_range[1])
    
    def forward(self, input: Tensor | np.ndarray) -> Image.Image:
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
        return Image.fromarray(output.numpy().astype(dtype=np.uint8), 'RGB')

class ToNumpy(Transform[Tensor | Image.Image, np.ndarray]):
    def forward(self, input: Tensor | Image.Image) -> np.ndarray:
        if isinstance(input, Tensor): return input.numpy()
        elif isinstance(input, Image.Image): return np.array(input)
        else: raise ValueError(f'Unsupported input type: {type(input)}')

class ConvertDtype(Transform[Tensor, Tensor]):
    def __init__(
        self,
        new_dtype: DTypeLike = float32
    ) -> None:
        super().__init__()
        self.new_dtype = new_dtype
    
    def forward(self, input: Tensor) -> Tensor:
        return input.to(dtype=self.new_dtype)

class ChangeDevice(Transform[Tensor, Tensor]):
    def __init__(
        self,
        new_device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        super().__init__(new_device)
        self.new_device = new_device
        
    def forward(self, input: Tensor) -> Tensor:
        return input.to(self.new_device)
    
class ToCPU(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input: Tensor) -> Tensor:
        return input.contiguous().cpu()
    
class ToCUDA(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.contiguous().cuda()
    
class Cast(Transform[Tensor, Tensor]):
    def __init__(
        self,
        new_device: Literal['cpu', 'cuda'] | None = None,
        new_dtype: DTypeLike | None = None
    ) -> None:
        super().__init__(new_device)
        self.new_device = new_device
        self.new_dtype = new_dtype
        
    def forward(self, input: Tensor) -> Tensor:
        dtype  = input.dtype  if self.new_dtype  is None else self.new_dtype
        device = input.device if self.new_device is None else self.new_device
        return input.to(device, dtype)

class ToContiguous(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input: Tensor) -> Tensor:
        return input.contiguous()

