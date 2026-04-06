from typing import Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision.procedurals import Generator
from nectarml.typing import DTypeLike, float32, uint8
from nectarml.vision.transforms.common import gradient_mask, lerp

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
        
    def forward(self) -> Tensor:
        out = Tensor(self.color, dtype=uint8)
        out = out.view((1, 3, 1, 1)).expand((1, 3)+self.size)
        return out.to(self.device, self.dtype)

class Gradient(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        mode: Literal[
            'horizontal', 'vertical', 'radial', 'elliptical'
        ] = 'horizontal',
        radial_method: Literal['corners', 'edges'] = 'corners',
        falloff_power: float = 1.0,
        invert: bool = False,
        color1: tuple[int, int, int] = (0, 0, 0),
        color2: tuple[int, int, int] = (255, 255, 255),
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        self.mode = mode
        self.radial_method = radial_method
        self.falloff_power = falloff_power
        self.invert = invert
        self.color1 = color1
        self.color2 = color2
        
    def _generate(self) -> Tensor:
        mask = gradient_mask(
            shape=(1, 1)+self.size, 
            mode=self.mode,
            radial_method=self.radial_method,
            invert=self.invert
        )**self.falloff_power
        
        arr = np.ones(self.size, dtype=uint8)
        values1 = [arr*self.color1[0], arr*self.color1[1], arr*self.color1[2]]
        values2 = [arr*self.color2[0], arr*self.color2[1], arr*self.color2[2]]
        
        d1 = np.stack(values1).astype(self.dtype)
        d2 = np.stack(values2).astype(self.dtype)
        
        col1, col2 = Tensor(d1, dtype=self.dtype), Tensor(d2, dtype=self.dtype)
        return lerp(col1, col2, mask)

    def forward(self) -> Tensor:
        return self._generate()


