from typing import Literal

import numpy as np
from PIL import Image

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.vision.transforms import Transform

class ToTensor(Transform[np.ndarray | Image.Image, Tensor]):
    def __init__(
        self, 
        device: Literal['auto', 'cpu', 'cuda'] = 'auto',
        dtype: DTypeLike = float32,
        batch_dim: bool = True
    ) -> None:
        super().__init__(device)
        self.dtype = dtype
        self.batch_dim = batch_dim
    
    def forward(self, input: np.ndarray | Image.Image) -> Tensor:
        if isinstance(input, Image.Image):
            data = np.array(input).astype(self.dtype)
        elif isinstance(input, np.ndarray): data = input
        else: raise ValueError(f'Unsupported input type: {type(input)}')
        
        output = Tensor(data, dtype=self.dtype, device=self.device)
        output = output.permute((2, 0, 1))
        if self.batch_dim: output = output.unsqueeze(dim=0)
        return output

class ToPIL(Transform[Tensor | np.ndarray, Image.Image]):
    def __init__(
        self,
        normalize: bool = False,
        value_range: tuple[int, int] = (0, 255)
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.value_range = value_range
    
    def forward(self, input: Tensor | np.ndarray) -> Image.Image:
        assert input.ndim in [3, 4], \
            'ToPIL expects input to be 3D ([C, H, W]) or 4D ([B, C, H, W])'
        if input.ndim == 4:
            assert input.shape[0] == 1, \
                'ToPIL expects input to have only 1 batch ([1, H, W])'
            input.squeeze(0) 
        
        x = np.array(input.data, input.dtype)
        x.transpose()
        
        if isinstance(input, Tensor):
            input = input.permute((1, 2, 0))
            if normalize: input = _normalize(input, value_range)
            return Image.fromarray(input.numpy().astype(dtype=uint8), 'RGB')
        elif isinstance(input, np.ndarray):
            input = input.transpose(1, 2, 0)
        else: raise ValueError(f'Unsupported input type: {type(input)}')

class ToNumpy(Transform[Tensor | Image.Image, np.ndarray]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor | Image.Image) -> np.ndarray:
        pass

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
        dtype = input.dtype if self.new_dtype is None else self.new_dtype
        device = input.device if self.new_device is None else self.new_device
        return input.to(device, dtype)

class Permute(Transform[Tensor, Tensor]):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims
    
    def forward(self, input: Tensor) -> Tensor:
        return input.permute(self.dims)
    
class Clamp(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        min_value: float | None = None,
        max_value: float | None = None
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, input: Tensor) -> Tensor:
        return input.clamp(self.min_value, self.max_value)

