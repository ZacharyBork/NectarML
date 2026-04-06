from typing import Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision.procedurals import Generator
from nectarml.typing import DTypeLike, float32, uint8

class SdfCreate(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        sdf_type: Literal[
            'circle', 'square', 'triangle', 'pentagon', 'hexagon', 'octagon',
            'hexagram'
        ] = 'circle',
        radius: float = 0.5,
        center: tuple[float, float] = (0.5, 0.5),
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        '''
        Massive thank you to Inigo Quilez for the SDF algorithms:
        - https://iquilezles.org/articles/distfunctions2d/
        '''
        super().__init__(size, dtype, device)
        self.sdf_type = sdf_type
        self.radius = radius * 0.5
        self.center = center

    def _circle(self) -> np.ndarray:
        dist = np.sqrt((self._xs - self._cx) ** 2 + (self._ys - self._cy) ** 2)
        return dist - self.radius * np.minimum(self._W, self._H)
        
    def _square(self) -> np.ndarray:
        half_size = self.radius * np.minimum(self._W, self._H)
        dx = np.abs(self._xs - self._cx) - half_size
        dy = np.abs(self._ys - self._cy) - half_size

        outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
        inside = np.minimum(np.maximum(dx, dy), 0)

        return outside + inside
        
    def _triangle(self) -> np.ndarray:
        k = np.sqrt(3.0)
        r = self.radius * np.minimum(self._W, self._H)

        dx = np.abs(self._xs - self._cx) - r
        dy = self._ys - self._cy + r / k
        
        mask = (dx + k * dy) > 0.0
        new_dx, new_dy = (dx - k * dy) / 2.0, (-k * dx - dy) / 2.0
        dx, dy = np.where(mask, new_dx, dx), np.where(mask, new_dy, dy)
        
        dx -= np.clip(dx, -2.0 * r, 0.0)
        return -np.sqrt(dx**2 + dy**2) * np.sign(dy)
    
    def _pentagon(self) -> np.ndarray:
        kx, ky, kz = 0.809016994, 0.587785252, 0.726542528
        r = self.radius * np.minimum(self._W, self._H)

        px = np.abs(self._xs - self._cx)
        py = self._ys - self._cy

        d1 = np.minimum(-kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * (-kx)
        py -= 2.0 * d1 * ky

        d2 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d2 * kx
        py -= 2.0 * d2 * ky

        px -= np.clip(px, -r*kz, r*kz)
        py -= r

        return np.sqrt(px**2 + py**2) * np.sign(py)
        
    def _hexagon(self) -> np.ndarray:
        kx, ky, kz = -0.866025404, 0.5, 0.577350269
        r = self.radius * np.minimum(self._W, self._H)
        
        px = np.abs(self._xs - self._cx)
        py = np.abs(self._ys - self._cy)
        
        d1 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * kx
        py -= 2.0 * d1 * ky
        
        px -= np.clip(px, -kz*r, kz*r)
        py -= r
        
        return np.sqrt(px**2 + py**2) * np.sign(py)
            
    def _octagon(self) -> np.ndarray:
        kx, ky, kz = -0.9238795325, 0.3826834323, 0.4142135623
        r = self.radius * np.minimum(self._W, self._H)
        
        px = np.abs(self._xs - self._cx)
        py = np.abs(self._ys - self._cy)
        
        d1 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * kx
        py -= 2.0 * d1 * ky
        
        d2 = np.minimum(-kx*px + ky*py, 0.0)
        px -= 2.0 * d2 * -kx
        py -= 2.0 * d2 * ky
        
        px -= np.clip(px, -kz*r, kz*r)
        py -= r
        
        return np.sqrt(px**2 + py**2) * np.sign(py)
                
    def _hexagram(self) -> np.ndarray:
        pass
                
    def _generate(self) -> np.ndarray:
        self._H, self._W, = self.size
        self._cx, self._cy = self.center[0] * self._H, self.center[1] * self._W
        self._ys, self._xs = np.mgrid[0:self.size[0], 0:self.size[1]]

        match self.sdf_type:
            case 'circle':   return self._circle()
            case 'square':   return self._square()
            case 'triangle': return self._triangle()
            case 'pentagon': return self._pentagon()
            case 'hexagon':  return self._hexagon()
            case 'octagon':  return self._octagon()
            case 'hexagram': return self._hexagram()
            case _: raise ValueError(f'Invalid sdf type: {self.sdf_type}')
        
    def forward(self) -> Tensor:
        arr = self._generate()
        return Tensor(arr.astype(self.dtype)).unsqueeze(0)

    
