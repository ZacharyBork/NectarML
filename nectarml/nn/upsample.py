from __future__ import annotations

import math
import warnings
from   typing import Literal

from nectarml.core          import Tensor
from nectarml.nn.module     import Module
from nectarml.nn.functional import upsample

class Upsample(Module):
    def __init__(
        self: Upsample,
        size:         int   | tuple[int,   ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        a:                     float = -0.75,
        align_corners:          bool = False,
        recompute_scale_factor: bool = False,
        preserve_aspect_ratio:  bool = False
    ) -> None:
        '''Upsamples (or downsamples) tensors along their spatial dimensions.

        ### Output size

        Output size of the upsample opereration can be defined in one of two
        ways:

        1. `size`: Directly defining the size of the spatial dimensions as a:
            - A Single integer (L) for 3-dimension tensors.
            - A tuple (H: int, W: int) for 4-dimension tensors.
            - A tuple (D: int, H: int, W: int) for 5-dimension tensors.
            
        2. `scale_factor`: This acts as a multiplier to the spatial dimensions
            of the input tensor. So if you input a tensor with shape 
            (1, 3, 256, 256) and set `scale_factor` to (2.0, 2.0), the 
            output tensor would have shape (1, 3, 512, 512). `scale_factor` is
            defined as:
            
            - A Single float (L) for 3-dimension tensors.
            - A tuple (H: float, W: float) for 4-dimension tensors.
            - A tuple (D: float, H: float, W: float) for 5-dimension tensors.

        NOTE: If both `size` and `scale_factor` are provided, Upsample will use 
        `scale_factor`.

        ### Upsampling modes
        
        Certain upsampling modes are only valid for input tensors with certain
        shapes.
        
        For 3/4/5-dimension tensors:
            - 'nearest' : Performs nearest neighbour upsampling.
            
        For 3-dimension (B, C, L) tensors only:
            - 'linear' : Performs linear upsampling along 1 spatial dimension.
            
        For 4-dimension (B, C, H, W) tensors only:
            - 'bilinear' : Bilinear upsampling along H and W dimensions.
            - 'bicubic'  : Bicubic upsampling along H and W dimensions.

        For 5-dimension (B, C, D, H, W) tensors only:
            - 'trilinear' : Performs trilinear upsampling along 3 spatial
                            dimensions.

        Args:
            size          : The desired output size.
            scale_factor  : The scale factor for the upsample.
            mode          : The upsampling algorithm to use. Options are 
                            ['nearest', 'linear', 'bilinear', 'bicubic', 
                            'trilinear']
            align_corners : If True, the corner pixels of the input and output
                            tensor will be aligned.
            recompute_scale_factor : If True, scale factor will be recomputed
                                     by rounding to the closest pixel-aligned
                                     value from the first input. 
            preserve_aspect_ratio  : If True, the aspect ratio of the output 
                                     tensor will match that of the input.
        '''
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
    
    
    
