from typing import Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision.procedurals import Generator
from nectarml.typing import DTypeLike, float32

class Checkerboard(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        tiling: int | tuple[int, int] = (10, 10),
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        self.tiling = (tiling, tiling) if isinstance(tiling, int) else tiling
        
    def _generate(self) -> np.ndarray:
        y = np.linspace(0, self.tiling[0], self.size[0], endpoint=False)
        x = np.linspace(0, self.tiling[1], self.size[1], endpoint=False)        
        xx, yy = np.meshgrid(x, y)
        pattern = (np.floor(xx).astype(int) + np.floor(yy).astype(int)) % 2
        return pattern.astype(np.float32)

    def forward(self) -> Tensor:
        arr = self._generate()
        return Tensor(arr).unsqueeze(0).unsqueeze(0)


