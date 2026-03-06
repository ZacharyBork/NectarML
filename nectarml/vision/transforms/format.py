from typing import Literal

from PIL import Image

from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

class ToTensor(Transform):
    def __init__(
        self, 
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        super().__init__(device)
    
    def forward(self, input: Tensor) -> Tensor:
        pass

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

