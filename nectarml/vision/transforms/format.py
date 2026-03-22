from typing import Literal

import numpy as np
from PIL import Image

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
from nectarml.vision.transforms import Transform

class ToTensor(Transform):
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

class ToPIL(Transform):
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

class ToNumpy(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ConvertDtype(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Permute(Transform):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

