from __future__ import annotations

import math
import warnings
from   typing import Literal

from nectarml.core       import Tensor
from nectarml.nn.module  import Module
from nectarml.functional import upsample

class Upsample(Module):
    def __init__(
        self: Upsample,
        size:         int   | tuple[int, ...]   | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        a:                     float = -0.75,
        align_corners:          bool = False,
        recompute_scale_factor: bool = False,
        preserve_aspect_ratio:  bool = False
    ) -> None:
        super().__init__()
        self.mode = mode
        self.a    = a
        
        self.align_corners          = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.preserve_aspect_ratio  = preserve_aspect_ratio
        
        self._validated       = False
        self._is_scale_factor = True
                
        self.input_dims:  tuple[int, ...] = None
        self.output_dims: tuple[int, ...] = None
        
        self._init_scaling(size, scale_factor)
        
    ### INIT ###
        
    def _init_scaling(
        self:         Upsample,
        size:         int   | tuple[int, ...]   | None = None,
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
        
    ### UTILS ###
    
    def _init_scale_from_input(self: Upsample, x: Tensor) -> None:
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
                    int(math.floor(dim * scale)) / dim 
                    for dim, scale in zip(x.shape[2:], self._scale))
                        
        self._validated = True
        
    def _compute_dimensions(self: Upsample, x: Tensor) -> None:
        if self.input_dims is not None and self.output_dims is not None: return
        self.input_dims  = x.shape[2:]
        self.output_dims = tuple(
            int(math.floor(s * f)) 
            for s, f in zip(self.input_dims, self._scale))
        
    ### FORWARD ###
        
    def forward(self: Upsample, x: Tensor) -> Tensor:
        self._init_scale_from_input(x)
        self._compute_dimensions(x)
        if self._is_scale_factor:
            return upsample(
                x, scale_factor=self._scale, mode=self.mode, 
                a=self.a, align_corners=self.align_corners,
                preserve_aspect_ratio=self.preserve_aspect_ratio)
        else: 
            return upsample(
                x, size=self._scale, mode=self.mode, 
                a=self.a, align_corners=self.align_corners,
                preserve_aspect_ratio=self.preserve_aspect_ratio)
    
    
    
