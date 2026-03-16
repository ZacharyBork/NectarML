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
        if self.batch_dim: output = output.reshape((1,) + output.shape)
        return output

class ToPIL(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ToNumpy(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ConvertDtype(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Permute(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

