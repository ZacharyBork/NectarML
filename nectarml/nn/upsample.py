import math
import warnings
from typing import Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32
import nectarml.nn as nn

class Upsample(nn.Module):
    def __init__(
        self,
        size: int | tuple[int, int] | tuple[int, int, int] | None = None,
        scale_factor: float | tuple[float, float] | tuple[float, float, float]\
            | None = None,
        mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.align_corners = align_corners or False
        self.recompute_scale_factor = recompute_scale_factor or False
        self._validated = False
        self._is_scale_factor = True
        
        self.input_dims: tuple[int, ...] = None
        self.output_dims: tuple[int, ...] = None
        
        self._init_scaling(size, scale_factor)
        self._init_sampling_op(mode)
        
    ### INIT ###
        
    def _init_scaling(
        self,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None
    ) -> None:
        if size is None and scale_factor is None:
            raise ValueError(
                'Upsample must be initialized with either a "size" or a '
                '"scale_factor".')
        if size is not None and scale_factor is not None:
            warnings.warn(
                'Upsample initialized with both "size" and "scale_factor". '
                'Defaulting to "scale_factor".')
        
        self._is_scale_factor = scale_factor is not None
        self._scale = scale_factor if self._is_scale_factor else size
        
    def _init_sampling_op(
        self, 
        mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
    ) -> None:
        match mode:
            case 'nearest':   self.resample = self._nearest
            case 'linear':    self.resample = self._linear
            case 'bilinear':  self.resample = self._bilinear
            case 'bicubic':   self.resample = self._bicubic
            case 'trilinear': self.resample = self._trilinear
        
    ### UTILS ###
    
    def _init_scale_from_input(self, x: Tensor) -> None:
        if self._validated: return
        spatial_dims = len(x.shape[2:])
        
        if isinstance(self._scale, tuple):
            scale_dims = len(self._scale)
            assert spatial_dims == scale_dims, (
                f'Upsampling scale dims ({scale_dims}) does not match number '
                f'of spatial dims in input Tensor ({spatial_dims}).')
        elif isinstance(self._scale, int | float):
            self._scale = (self._scale,) * spatial_dims
            
        if self.recompute_scale_factor and self._is_scale_factor:
            if self.recompute_scale_factor:
                self._scale = tuple(
                    int(np.floor(dim * scale)) / dim 
                    for dim, scale in zip(x.shape[2:], self._scale))
                        
        self._validated = True
        
    def _compute_dimensions(self, x: Tensor) -> None:
        if self.input_dims is not None and self.output_dims is not None: return
        self.input_dims = x.shape[2:]
        self.output_dims = tuple(
            int(np.floor(s * f)) for s, f in zip(self.input_dims, self._scale))
        
    ### COORDINATE MAPPING ###
    
    def _compute_input_coordinates(
        self, 
        input_size: int,
        output_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_coords = np.arange(output_size)
        mapped = out_coords * (input_size / output_size)
        floor = np.floor(mapped).astype(int)
        ceil = np.minimum(floor + 1, input_size - 1)
        weight = mapped - floor
        return floor, ceil, weight
    
    ### SAMPLING OPS ###
        
    def _nearest(self, x: Tensor) -> Tensor:
        B, C = x.shape[:2]

        indices = [
            np.floor(np.arange(out) * (in_dim / out)).astype(int)
            for out, in_dim in zip(self.output_dims, self.input_dims)]

        new_data = x.data[:, :][np.ix_(np.arange(B), np.arange(C), *indices)]
        out = x._build_output_tensor(new_data, children=(x,))
                
        def _backward():
            if x.requires_grad:
                grad = np.zeros_like(x.data)
                values = np.ix_(np.arange(B), np.arange(C), *indices)
                np.add.at(grad, values, out.grad)
                x.grad += grad
        
        out._backward = _backward
        return out
        
    def _linear(self, x: Tensor) -> Tensor:
        B, C = x.shape[:2]
        in_l = x.shape[2]
        out_l = int(np.floor(in_l * self._scale[0]))
        
        floor, ceil, weight = self._compute_input_coordinates(in_l, out_l)
        
        left = x.data[:, :, floor]
        right = x.data[:, :, ceil]
        new_data = (1 - weight) * left + weight * right
        out = x._build_output_tensor(new_data, children=(x,))
        
        def _backward():
            if x.requires_grad:
                pass
        out._backward = _backward
        return out
    
    def _bilinear(self, x: Tensor) -> Tensor:
        pass
    
    def _bicubic(self, x: Tensor) -> Tensor:
        pass
    
    def _trilinear(self, x: Tensor) -> Tensor:
        pass
        
    ### FORWARD ###
        
    def forward(self, x: Tensor) -> Tensor:
        self._init_scale_from_input(x)
        self._compute_dimensions(x)
        return self.resample(x)
    
    
