from typing import Literal

import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.creation import ones_like
from nectarml.vision.procedurals import Generator
from nectarml.typing import DTypeLike, float32, uint8
from nectarml.vision.transforms.common import lerp

class SdfCreate(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        sdf_type: Literal[
            'circle', 'square', 'triangle', 'pentagon', 'hexagon', 'octagon',
            'hexagram', 'pentagram'
        ] = 'circle',
        radius: float = 0.5,
        center: tuple[float, float] = (0.5, 0.5),
        angle: float = 0.0,
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
        self.angle = angle

    def _circle(self) -> np.ndarray:
        dist = np.sqrt((self._xs) ** 2 + (self._ys) ** 2)
        return dist - self.radius * np.minimum(self._W, self._H)
        
    def _square(self) -> np.ndarray:
        dx = np.abs(self._xs) - self._r
        dy = np.abs(self._ys) - self._r

        outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
        inside = np.minimum(np.maximum(dx, dy), 0)

        return outside + inside
        
    def _triangle(self) -> np.ndarray:
        k = np.sqrt(3.0)

        dx = np.abs(self._xs) - self._r
        dy = self._ys + self._r / k
        
        mask = (dx + k * dy) > 0.0
        new_dx, new_dy = (dx - k * dy) / 2.0, (-k * dx - dy) / 2.0
        dx, dy = np.where(mask, new_dx, dx), np.where(mask, new_dy, dy)
        
        dx -= np.clip(dx, -2.0 * self._r, 0.0)
        return -np.sqrt(dx**2 + dy**2) * np.sign(dy)
    
    def _pentagon(self) -> np.ndarray:
        kx, ky, kz = 0.809016994, 0.587785252, 0.726542528

        px = np.abs(self._xs)
        py = self._ys

        d1 = np.minimum(-kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * (-kx)
        py -= 2.0 * d1 * ky

        d2 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d2 * kx
        py -= 2.0 * d2 * ky

        px -= np.clip(px, -self._r*kz, self._r*kz)
        py -= self._r

        return np.sqrt(px**2 + py**2) * np.sign(py)
        
    def _hexagon(self) -> np.ndarray:
        kx, ky, kz = -0.866025404, 0.5, 0.577350269
        
        px = np.abs(self._xs)
        py = np.abs(self._ys)
        
        d1 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * kx
        py -= 2.0 * d1 * ky
        
        px -= np.clip(px, -kz*self._r, kz*self._r)
        py -= self._r
        
        return np.sqrt(px**2 + py**2) * np.sign(py)
            
    def _octagon(self) -> np.ndarray:
        kx, ky, kz = -0.9238795325, 0.3826834323, 0.4142135623
        
        px = np.abs(self._xs)
        py = np.abs(self._ys)
        
        d1 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * kx
        py -= 2.0 * d1 * ky
        
        d2 = np.minimum(-kx*px + ky*py, 0.0)
        px -= 2.0 * d2 * -kx
        py -= 2.0 * d2 * ky
        
        px -= np.clip(px, -kz*self._r, kz*self._r)
        py -= self._r
        
        return np.sqrt(px**2 + py**2) * np.sign(py)
                
    def _hexagram(self) -> np.ndarray:
        kx, ky, kz, kw = -0.5, 0.8660254038, 0.5773502692, 1.7320508076
        self._r *= 0.5
        
        px = np.abs(self._xs)
        py = np.abs(self._ys)

        d1 = np.minimum(kx*px + ky*py, 0.0)
        px -= 2.0 * d1 * kx
        py -= 2.0 * d1 * ky
        
        d2 = np.minimum(ky*px + kx*py, 0.0)
        px -= 2.0 * d2 * ky
        py -= 2.0 * d2 * kx
        
        px -= np.clip(px, self._r*kz, self._r*kw)
        py -= self._r
        
        return np.sqrt(px**2 + py**2) * np.sign(py)
              
    def _pentagram(self) -> np.ndarray:
        k1x, k2x = 0.809016994, 0.309016994
        k1y, k2y = 0.587785252, 0.951056516
        k1z = 0.726542528
        
        px = np.abs(self._xs)
        py = self._ys
        
        d1 = np.maximum(k1x*px + -k1y*py, 0.0)
        px -= 2.0 * d1 * k1x
        py -= 2.0 * d1 * -k1y
        
        d2 = np.maximum(-k1x*px + -k1y*py, 0.0)
        px -= 2.0 * d2 * -k1x
        py -= 2.0 * d2 * -k1y
        
        px = np.abs(px)
        py -= self._r
        
        t = np.clip(px*k2x + py*-k2y, 0.0, k1z*self._r)
        qx, qy = (px - k2x * t), (py - -k2y * t)
        return np.sqrt(qx**2 + qy**2) * np.sign(py*k2x - px*-k2y)

    def _generate(self) -> np.ndarray:
        self._H, self._W, = self.size
        self._r = self.radius * np.minimum(self._W, self._H)
        self._ys, self._xs = np.mgrid[0:self.size[0], 0:self.size[1]]
        
        _angle = np.radians(self.angle)
        rot = np.array([
            [ np.cos(_angle), np.sin(_angle)],
            [-np.sin(_angle), np.cos(_angle)]])

        cx, cy = self.center[0] * self._H, self.center[1] * self._W
        xs_c, ys_c = self._xs - cx, self._ys - cy
        self._xs, self._ys = np.einsum(
            'ji, mni -> jmn', rot, np.dstack([xs_c, ys_c]))

        match self.sdf_type:
            case 'circle':    return self._circle()
            case 'square':    return self._square()
            case 'triangle':  return self._triangle()
            case 'pentagon':  return self._pentagon()
            case 'hexagon':   return self._hexagon()
            case 'octagon':   return self._octagon()
            case 'hexagram':  return self._hexagram()
            case 'pentagram': return self._pentagram()
            case _: raise ValueError(f'Invalid sdf type: {self.sdf_type}')
        
    def forward(self) -> Tensor:
        arr = self._generate()
        return Tensor(arr.astype(self.dtype)).unsqueeze(0)

class SdfCombine(Generator):
    def __init__(
        self, 
        method: Literal[
            'union', 'intersect', 'subtract', 'difference'
        ] = 'union',
        radius: float = 0.0
    ) -> None:
        '''
        Thank you to Inigo Quilez for the SDF combination functions:
        - https://iquilezles.org/articles/distfunctions/
        '''
        super().__init__(None, None, None)
        self.method = method
        self.radius = radius

    def generate(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.device == y.device, \
            f'SdfCombine requires all SDF Tensors to be on the same device, ' \
            f'but found two devices: {x.device} and {y.device}'
        assert x.shape == y.shape, \
            'SdfCombine requires all SDF Tensors to have the same shape.'
        match self.method:
            case 'union':
                if self.radius == 0.0: return F.minimum(x, y)
                h = F.clamp(0.5 + 0.5 * (y - x) / self.radius, 0.0, 1.0)
                return y + (x - y) * h - self.radius * h * (1.0 - h)
            case 'intersect':
                if self.radius == 0.0: return F.maximum(x, y)
                h = F.clamp(0.5 - 0.5 * (y - x) / self.radius, 0.0, 1.0)
                return y + (x - y) * h + self.radius * h * (1.0 - h)
            case 'subtract':
                if self.radius == 0.0: return F.maximum(x, -y)
                h = F.clamp(0.5 - 0.5 * (-y - x) / self.radius, 0.0, 1.0)
                return -y + (x - -y)*h + self.radius*h*(1.0 - h)
            case 'difference':
                if self.radius == 0.0: return -F.minimum(-x, -y)
                h = F.clamp(0.5 + 0.5 * (x - y) / self.radius, 0.0, 1.0)
                return y + (x - y)*h + self.radius*h*(1.0 - h)
            case _: raise ValueError(
                f'Combination method not valid: {self.method}')

class SdfToGray(Generator):
    def __init__(
        self, 
        iso_value: float = 0.0
    ) -> None:
        super().__init__(None, None, None)
        self.iso_value = iso_value

    def generate(self, sdf: Tensor) -> Tensor:
        max_value = sdf.max().item()
        norm = sdf / max_value
        iso = F.where(norm<self.iso_value, ones_like(norm), 0.0)
        return iso * max_value

class SdfColorRamp(Generator):
    def __init__(
        self, 
        color1: tuple[int, int, int] = (255, 0, 0),
        color2: tuple[int, int, int] = (0, 255, 0),
        contrast: float = 3.0
    ) -> None:
        super().__init__(None, None, None)
        self.color1 = color1
        self.color2 = color2
        self.contrast = contrast

    def generate(self, sdf: Tensor) -> Tensor:
        _min, _max = sdf.min().item(), sdf.max().item()
        spatial = (sdf.shape[-2], sdf.shape[-1])
        
        if _min == _max:
            if _min == 0.0: ramp = sdf
            ramp = sdf / _min
        else: ramp = (sdf - _min) / (_max - _min)
        
        color1 = Tensor(self.color1, dtype=uint8)
        color1 = color1.view((1, 3, 1, 1)).expand((1, 3)+spatial)
        
        color2 = Tensor(self.color2, dtype=uint8)
        color2 = color2.view((1, 3, 1, 1)).expand((1, 3)+spatial)
        
        out = lerp(color1, color2, ramp).to(sdf.device, sdf.dtype)
        out = (((out / _max) - 0.5) * self.contrast + 0.5) * _max
        return out.clamp(0.0, _max)


