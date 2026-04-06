from typing import Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision.procedurals import Generator
from nectarml.typing import DTypeLike, float32

class Solid(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        color: tuple[int, int, int] = (255, 0, 0),
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        self.color = color
        
    def _generate(self) -> np.ndarray:
        arr = np.ones(self.size, dtype=np.uint8)
        r, g, b = self.color
        return np.stack([arr*r, arr*g, arr*b])

    def forward(self) -> Tensor:
        arr = self._generate()
        return Tensor(arr).unsqueeze(0)




