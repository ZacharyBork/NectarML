from typing import Literal

import numpy as np
from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite, 
    NoiseType,
    FractalType,
    CellularDistanceFunction
)

from nectarml import typing
from nectarml.core import Tensor
from nectarml.vision.procedurals import Generator

class Noise(Generator):
    def __init__(
        self, 
        size: tuple[int, int] = (256, 256),
        noise_type: Literal[
            'perlin', 'simplex', 'voronoi', 'value', 'white'
        ] = 'perlin',
        fractal_type: Literal['none', 'fbm', 'ridged', 'pingpong'] = 'fbm',
        distance_metric: Literal[
            'euclidean', 'manhattan', 'hybrid', 'euclideansq'
        ] = 'euclidean',
        output_range: tuple[float, float] = (0.0, 1.0),
        scale: float = 1.0,
        frequency: float = 0.05,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int | None = None,
        dtype: typing.dtype = typing.float32, 
        device: typing.DeviceLikeType = 'cpu'
    ) -> None:
        super().__init__(size, dtype, device)
        match noise_type:
            case 'perlin':  self.noise_type = NoiseType.NoiseType_Perlin
            case 'simplex': self.noise_type = NoiseType.NoiseType_OpenSimplex2
            case 'worley':  self.noise_type = NoiseType.NoiseType_Cellular
            case 'voronoi': self.noise_type = NoiseType.NoiseType_Cellular
            case 'value':   self.noise_type = NoiseType.NoiseType_Value
            case 'white':   self.noise_type = None
            
        match fractal_type:
            case 'none':     self.fractal = FractalType.FractalType_None
            case 'fbm':      self.fractal = FractalType.FractalType_FBm
            case 'ridged':   self.fractal = FractalType.FractalType_Ridged
            case 'pingpong': self.fractal = FractalType.FractalType_PingPong
        
        d = CellularDistanceFunction
        match distance_metric:
            case 'euclidean': 
                self.metric = d.CellularDistanceFunction_Euclidean
            case 'manhattan': 
                self.metric = d.CellularDistanceFunction_Manhattan
            case 'hybrid': 
                self.metric = d.CellularDistanceFunction_Hybrid
            case 'euclideansq': 
                self.metric = d.CellularDistanceFunction_EuclideanSq

        self.output_range = output_range
        self.scale = scale
        self.frequency = frequency
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed

    def _generate(self) -> np.ndarray:
        fn = FastNoiseLite()
        if self.seed is not None: fn.seed = self.seed
        fn.noise_type                 = self.noise_type
        fn.frequency                  = self.frequency
        fn.fractal_type               = self.fractal
        fn.fractal_octaves            = self.octaves
        fn.fractal_gain               = self.persistence
        fn.fractal_lacunarity         = self.lacunarity
        fn.cellular_distance_function = self.metric
        
        yy, xx = np.meshgrid(
            np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        yy, xx = yy*self.scale, xx*self.scale
        
        result = np.vectorize(fn.get_noise)(xx, yy).astype(np.float32)
        return result

    def _map_output_range(self, input: Tensor) -> Tensor:
        _min, _max = input.min().item(), input.max().item() 
        _rmin, _rmax = self.output_range[0], self.output_range[1]
        if _min == _max:
            if _min == 0.0: return input
            return input / _min * _rmax
        else: return ((input - _min) * ((_rmax - _rmin) / (_max - _min)))

    def forward(self) -> Tensor:
        arr = self._generate()
        out = Tensor(arr.astype(self.dtype.numpy)).unsqueeze(0).unsqueeze(0)
        return self._map_output_range(out)
