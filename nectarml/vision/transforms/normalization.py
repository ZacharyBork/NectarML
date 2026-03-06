from typing import Literal

from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

class Normalize(Transform):
    def __init__(
        self,
        mean: list[float],
        std: list[float],
        inplace: bool = False,
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        super().__init__(device)
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def forward(self, input: Tensor) -> Tensor:
        if self.device == 'cuda':
            t_range = _nectarml.compute_tensor_range(
                input._data_ptr, input.size, nectarml.cuda.map_dtype(x.dtype))

class Denormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class MinMaxNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ToFloat(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class ToUint8(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

