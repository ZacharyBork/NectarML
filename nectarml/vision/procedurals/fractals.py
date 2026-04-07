from typing import Literal
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision.procedurals import Generator
from nectarml.typing import DTypeLike, float32, int32

@dataclass(frozen=True)
class TrapColors:
    DEEP_SEA = [
        [  0,   0,   0],
        [  0,   0, 128],
        [  0,  64, 192],
        [  0, 192, 255],
        [255, 255, 255]]

    FIRE = [
        [  0,   0,   0],
        [128,   0,   0],
        [255,  64,   0],
        [255, 192,   0],
        [255, 255, 255]]

    PSYCHEDELIC = [
        [  0,   0, 128],
        [128,   0, 255],
        [255,   0, 128],
        [255, 128,   0],
        [  0, 255, 128],
        [  0, 128, 255]]

    EARTH = [
        [ 10,   5,   3],
        [ 80,  40,  10],
        [160, 100,  40],
        [220, 180, 100],
        [240, 230, 200]]

    GOLD = [
        [  0,   0,   0],
        [ 64,  32,   0],
        [192, 128,   0],
        [255, 220, 100],
        [255, 255, 240]]

class ColorOrbitTrap(Generator):
    def __init__(
        self,
        preset: Literal[
            'deep_sea', 'fire', 'psychedelic', 'earth', 'gold'
        ] = 'deep_sea',
        custom_colors: list[list[int]] | None = None
    ) -> None:
        super().__init__(None, None, None)
        if custom_colors is None: 
            _colors = TrapColors()
            match preset:
                case 'deep_sea':    self.colors = _colors.DEEP_SEA
                case 'fire':        self.colors = _colors.FIRE
                case 'psychedelic': self.colors = _colors.PSYCHEDELIC
                case 'earth':       self.colors = _colors.EARTH
                case 'gold':        self.colors = _colors.GOLD
        else: self.colors = custom_colors

    def apply_colormap(self, input: Tensor) -> Tensor:
        colors = Tensor(np.array(self.colors, dtype=np.float32)) / 255.0
        colors = colors.to(input.device)
        n = len(colors) - 1
        
        idx = (input * n).to(dtype=int32).clamp(0, n-1)
        frac = (input * n) - idx
        
        c0 = colors[idx]
        c1 = colors[(idx+1).clamp(0, n)]
        
        interpolated = c0 + (c1 - c0) * frac.unsqueeze(-1)
        return interpolated.permute((0, 3, 1, 2))
        
    def generate(self, input: Tensor) -> Tensor:
        norm = input / input.max().item()
        out = self.apply_colormap(norm)
        return (out * 255).clamp(0.0, 255.0).to(input.device, input.dtype)

class Mandelbrot(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        center: tuple[float, float] = (0.5, 0.5),
        zoom: float = 1.0,
        angle: float = 0.0,
        bound: int = 2,
        power: float = 2.0,
        max_iterations: int = 50,
        trap: bool = False,
        trap_type: Literal[
            'point', 'cross', 'circle', 'box', 'cross+circle'
        ] = 'cross+circle',
        trap_radius: float = 1.5,
        trap_center: complex = 0.0 + 0.0j,
        custom_trap_fn: Callable[[complex], float] | None = None,
        dtype: DTypeLike = float32,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        self.center = center
        self.zoom = 1 / zoom
        self.angle = angle
        self.bound = bound
        self.power = power
        self.max_iterations = max_iterations
        self.trap = trap
        self.trap_type = trap_type
        self.trap_radius = trap_radius
        self.trap_center = trap_center
        self.custom_trap_fn = custom_trap_fn

    def _function(self, z: complex, p: float, c: complex) -> complex:
        return z**p + c

    def _iterate(self, c: complex) -> float:
        z, p = 0.0, self.power
        for iteration_number in range(self.max_iterations):
            if abs(z) >= self.bound:
                return iteration_number
            try: z = self._function(z, p, c)
            except (ValueError, ZeroDivisionError): z = c
        return 0.0

    def _trap(self, z: complex) -> float:
        match self.trap_type:
            case 'point': dist = abs(z - self.trap_center)
            case 'cross':
                dist = min(
                    abs(abs(z.real-self.trap_center.real) - self.trap_radius),
                    abs(abs(z.imag-self.trap_center.imag) - self.trap_radius))
            case 'circle':
                dist = abs(abs(z - self.trap_center) - self.trap_radius)
            case 'box':
                dx = abs(z.real-self.trap_center.real) - self.trap_radius
                dy = abs(z.imag-self.trap_center.imag) - self.trap_radius
                dist = max(min(dx, dy), 0.0)
            case 'cross+circle':
                dist = min(
                    abs(abs(z.real-self.trap_center.real) - self.trap_radius),
                    abs(abs(z.imag-self.trap_center.imag) - self.trap_radius),
                    abs(abs(z - self.trap_center) - self.trap_radius))
            case _: raise ValueError(
                f'trap_type not valid: {self.trap_type}')
        
        return dist

    def _iterate_trap(self, c: complex) -> float:
        z, p = 0.0, self.power
        min_dist = float('inf')
        escaped = False
        for _ in range(self.max_iterations):
            if abs(z) >= self.bound:
                escaped = True
                break
            try: z = self._function(z, p, c)
            except (ValueError, ZeroDivisionError): z = c
            
            if self.custom_trap_fn is None: dist = self._trap(z)
            else: dist = self.custom_trap_fn(z)
            if dist < min_dist: min_dist = dist
            
        if not escaped: return 0.0
        return min_dist

    def _generate(self) -> np.ndarray:
        y_domain = np.linspace(-2, 2, self.size[0], dtype=float32)
        y_domain = (y_domain + (self.center[0] - 0.5)) * self.zoom
        
        x_domain = np.linspace(-2, 2, self.size[1], dtype=float32)
        x_domain = (x_domain + (self.center[1] - 0.5)) * self.zoom

        xx, yy = np.meshgrid(x_domain, y_domain)
        grid = (xx + 1j * yy) * np.exp(1j * np.radians(self.angle))

        fn = self._iterate_trap if self.trap else self._iterate
        iteration_array = [
            [fn(grid[y, x]) for x in range(self.size[1])]
            for y in range(self.size[0])]
                
        output = np.array(iteration_array)
        if self.trap:
            mask = output > 0
            output[mask] = np.log1p(output[mask])
            lo = np.percentile(output[mask], 5)
            hi = np.percentile(output[mask], 95)
            output = np.clip((output - lo) / (hi - lo), 0.0, 1.0)
            return output * 255.0
        else: return (output / self.max_iterations) * 255
        
    def forward(self) -> Tensor:
        arr = self._generate()
        return Tensor(arr.astype(self.dtype)).unsqueeze(0)



