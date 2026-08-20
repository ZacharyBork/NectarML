import numpy as np
from scipy.ndimage import distance_transform_edt

from nectarml                    import typing
from nectarml.core               import Tensor
from nectarml.constants          import PI
from nectarml.vision.procedurals import Generator
from nectarml.functional         import lerp

class Checkerboard(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        tiling: int | tuple[int, int] = (10, 10),
        dtype: typing.dtype = typing.float32,
        device: typing.DeviceLikeType = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        self.tiling = (tiling, tiling) if isinstance(tiling, int) else tiling
        
    def _generate(self) -> np.ndarray:
        y = np.linspace(0, self.tiling[0], self.size[0], endpoint=False)
        x = np.linspace(0, self.tiling[1], self.size[1], endpoint=False)        
        xx, yy = np.meshgrid(x, y)
        pattern = (
            np.floor(xx).astype(np.int32)
          + np.floor(yy).astype(np.int32)
        ) % 2
        return (pattern * 255).astype(np.uint8)

    def forward(self) -> Tensor:
        arr = self._generate()
        return Tensor(arr.astype(self.dtype.numpy)).unsqueeze(0).unsqueeze(0)

class ChladniCymaticPatterns(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        m: int = 2,
        n: int = 3,
        scale: tuple[float, float] = (1.0, 1.0),
        threshold: float | None = 0.5,
        smoothing_width: float = 0.0,
        color1: tuple[int, int, int] = (255, 255, 255),
        color2: tuple[int, int, int] = (0, 0, 0),
        dtype: typing.dtype = typing.float32,
        device: typing.DeviceLikeType = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        self.m = m
        self.n = n
        self.scale = tuple([i*2*j for i, j in zip(scale, size)])
        self.threshold = threshold
        self.smoothing_width = smoothing_width
        self.color1 = color1
        self.color2 = color2
        
    def _smooth(self, array: np.ndarray) -> np.ndarray:
        dist_in = distance_transform_edt(array)
        dist_out = distance_transform_edt(1 - array)

        sdf = dist_in - dist_out
        t = np.clip(sdf / self.smoothing_width, -1.0, 1.0)
        t = t * 0.5 + 0.5
        return t * t * (3 - 2 * t)
        
    def _generate(self) -> np.ndarray:
        l = np.minimum(self.size[0], self.size[1])
        
        y = np.linspace(0, self.scale[0], self.size[0], dtype=np.float32)
        x = np.linspace(0, self.scale[1], self.size[1], dtype=np.float32)  
        xx, yy = np.meshgrid(x, y)
        
        pattern = np.cos((self.n*PI*xx)/l) * np.cos((self.m*PI*yy)/l) 
        pattern -= np.cos((self.m*PI*xx)/l) * np.cos((self.n*PI*yy)/l)
        out = (1.0 - np.abs(pattern).clip(0.0, 1.0))
        
        if self.threshold is not None:
            out = np.where(out > self.threshold, 1.0, 0.0)
            if self.smoothing_width > 0: 
                out = self._smooth(out)
        
        return out
    
    def forward(self) -> Tensor:
        arr = self._generate().astype(self.dtype.numpy)
        ramp = Tensor(arr, dtype=self.dtype).unsqueeze(0).unsqueeze(0)
        
        color1 = Tensor(self.color1, dtype=self.dtype)
        color1 = color1.view((1, 3, 1, 1)).expand((1, 3)+self.size)
        
        color2 = Tensor(self.color2, dtype=self.dtype)
        color2 = color2.view((1, 3, 1, 1)).expand((1, 3)+self.size)
        
        return lerp(color1, color2, 1-ramp)


