from __future__ import annotations

import builtins

from nectarml.core          import Tensor
from nectarml.nn.module     import Module
from nectarml.nn.functional import pad

### CONSTANT ###

class ConstantPad1d(Module):
    def __init__(
        self:    ConstantPad1d, 
        padding: builtins.int | tuple[builtins.int, builtins.int],
        value:   builtins.float = 0.0
    ) -> None:
        '''Applies constant padding to 3-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        constant value. Only valid for 3-dimensional tensors (B, C, L).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on both sides, or a tuple of
                      2 integers, in which case it will be treated as 
                      (start, end).
            fill    : The value (0-1) to fill the padded area with.
        '''
        super().__init__()
        self.padding = padding
        self.value   = value
        
    def forward(self: ConstantPad1d, x: Tensor) -> Tensor:
        assert x.ndim == 3, (
            'ConstantPad1d only accepts input Tensors with 3 dimensions '
            '([B, C, L]).')
        return pad(x, self.padding, 'constant', value=self.value)
    
class ConstantPad2d(Module):
    def __init__(
        self:    ConstantPad2d, 
        padding: builtins.int | tuple[builtins.int, ...],
        value:   builtins.float = 0.0
    ) -> None:
        '''Applies constant padding to 4-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        constant value. Only valid for 4-dimensional tensors (B, C, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 2 
                      integers, in which case it will be treated as 
                      (LR, TB), or a tuple of 4 integers for (L, R, T, B).
            fill    : The value (0-1) to fill the padded area with.
        '''
        super().__init__()
        self.padding = padding
        self.value   = value
        
    def forward(self: ConstantPad2d, x: Tensor) -> Tensor:
        assert x.ndim == 4, (
            'ConstantPad2d only accepts input Tensors with 4 dimensions '
            '([B, C, H, W]).')
        return pad(x, self.padding, 'constant', value=self.value)

class ConstantPad3d(Module):
    def __init__(
        self:    ConstantPad3d, 
        padding: builtins.int | tuple[builtins.int, ...],
        value:   builtins.float = 0.0
    ) -> None:
        '''Applies constant padding to 5-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        constant value. Only valid for 5-dimensional tensors (B, C, D, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 6 
                      integers for (L, R, T, B, F, B).
            fill    : The value (0-1) to fill the padded area with.
        '''
        super().__init__()
        self.padding = padding
        self.value   = value
        
    def forward(self: ConstantPad3d, x: Tensor) -> Tensor:
        assert x.ndim == 5, (
            'ConstantPad3d only accepts input Tensors with 5 dimensions '
            '([B, C, D, H, W]).')
        return pad(x, self.padding, 'constant', value=self.value)

### REFLECTION ###

class ReflectionPad1d(Module):
    def __init__(
        self:    ReflectionPad1d, 
        padding: builtins.int | tuple[builtins.int, builtins.int]
    ) -> None:
        '''Applies reflection padding to 3-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        values mirrored from the original edge pixels. Only valid for
        3-dimensional tensors (B, C, L).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on both sides, or a tuple of
                      2 integers, in which case it will be treated as 
                      (start, end).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ReflectionPad1d, x: Tensor) -> Tensor:
        assert x.ndim == 3, (
            'ReflectionPad1d only accepts input Tensors with 3 dimensions '
            '([B, C, L]).')
        return pad(x, self.padding, 'reflect')
    
class ReflectionPad2d(Module):
    def __init__(
        self:    ReflectionPad2d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies reflection padding to 4-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        values mirrored from the original edge pixels. Only valid for 
        4-dimensional tensors (B, C, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 2 
                      integers, in which case it will be treated as 
                      (LR, TB), or a tuple of 4 integers for (L, R, T, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ReflectionPad2d, x: Tensor) -> Tensor:
        assert x.ndim == 4, (
            'ReflectionPad2d only accepts input Tensors with 4 dimensions '
            '([B, C, H, W]).')
        return pad(x, self.padding, 'reflect')

class ReflectionPad3d(Module):
    def __init__(
        self:    ReflectionPad3d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies reflection padding to 5-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        values mirrored from the original edge pixels. Only valid for 
        5-dimensional tensors (B, C, D, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 6 
                      integers for (L, R, T, B, F, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ReflectionPad3d, x: Tensor) -> Tensor:
        assert x.ndim == 5, (
            'ReflectionPad3d only accepts input Tensors with 5 dimensions '
            '([B, C, D, H, W]).')
        return pad(x, self.padding, 'reflect')

### REPLICATION ###

class ReplicationPad1d(Module):
    def __init__(
        self:    ReplicationPad1d, 
        padding: builtins.int | tuple[builtins.int, builtins.int]
    ) -> None:
        '''Applies replication padding to 3-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        the value of the original edge pixels, replicated outward. Only valid 
        for 3-dimensional tensors (B, C, L).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on both sides, or a tuple of
                      2 integers, in which case it will be treated as 
                      (start, end).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ReplicationPad1d, x: Tensor) -> Tensor:
        assert x.ndim == 3, (
            'ReplicationPad1d only accepts input Tensors with 3 dimensions '
            '([B, C, L]).')
        return pad(x, self.padding, 'replicate')
    
class ReplicationPad2d(Module):
    def __init__(
        self:    ReplicationPad2d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies replication padding to 4-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        the value of the original edge pixels, replicated outward. Only valid 
        for 4-dimensional tensors (B, C, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 2 
                      integers, in which case it will be treated as 
                      (LR, TB), or a tuple of 4 integers for (L, R, T, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ReplicationPad2d, x: Tensor) -> Tensor:
        assert x.ndim == 4, (
            'ReplicationPad2d only accepts input Tensors with 4 dimensions '
            '([B, C, H, W]).')
        return pad(x, self.padding, 'replicate')

class ReplicationPad3d(Module):
    def __init__(
        self:    ReplicationPad3d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies replication padding to 5-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        the value of the original edge pixels, replicated outward. Only
        valid for 5-dimensional tensors (B, C, D, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 6 
                      integers for (L, R, T, B, F, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ReplicationPad3d, x: Tensor) -> Tensor:
        assert x.ndim == 5, (
            'ReplicationPad3d only accepts input Tensors with 5 dimensions '
            '([B, C, D, H, W]).')
        return pad(x, self.padding, 'replicate')

### CIRCULAR ###

class CircularPad1d(Module):
    def __init__(
        self:    CircularPad1d, 
        padding: builtins.int | tuple[builtins.int, builtins.int]
    ) -> None:
        '''Applies circular padding to 3-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        values wrapped from the opposite side of the tensor. Only valid for
        3-dimensional tensors (B, C, L).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on both sides, or a tuple of
                      2 integers, in which case it will be treated as 
                      (start, end).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: CircularPad1d, x: Tensor) -> Tensor:
        assert x.ndim == 3, (
            'CircularPad1d only accepts input Tensors with 3 dimensions '
            '([B, C, L]).')
        return pad(x, self.padding, 'circular')
    
class CircularPad2d(Module):
    def __init__(
        self:    CircularPad2d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies circular padding to 4-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        values wrapped from the opposite side of the tensor. Only valid for
        4-dimensional tensors (B, C, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 2 
                      integers, in which case it will be treated as 
                      (LR, TB), or a tuple of 4 integers for (L, R, T, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: CircularPad2d, x: Tensor) -> Tensor:
        assert x.ndim == 4, (
            'CircularPad2d only accepts input Tensors with 4 dimensions '
            '([B, C, H, W]).')
        return pad(x, self.padding, 'circular')

class CircularPad3d(Module):
    def __init__(
        self:    CircularPad3d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies circular padding to 5-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        values wrapped from the opposite side of the tensor. Only valid for 
        5-dimensional tensors (B, C, D, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 6 
                      integers for (L, R, T, B, F, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: CircularPad3d, x: Tensor) -> Tensor:
        assert x.ndim == 5, (
            'CircularPad3d only accepts input Tensors with 5 dimensions '
            '([B, C, D, H, W]).')
        return pad(x, self.padding, 'circular')
    
### ZERO ###

class ZeroPad1d(Module):
    def __init__(
        self:    ZeroPad1d, 
        padding: builtins.int | tuple[builtins.int, builtins.int]
    ) -> None:
        '''Applies zero padding to 3-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        zeros. Only valid for 3-dimensional tensors (B, C, L).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on both sides, or a tuple of
                      2 integers, in which case it will be treated as 
                      (start, end).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ZeroPad1d, x: Tensor) -> Tensor:
        assert x.ndim == 3, (
            'ZeroPad1d only accepts input Tensors with 3 dimensions '
            '([B, C, L]).')
        return pad(x, self.padding, 'constant', value=0)
    
class ZeroPad2d(Module):
    def __init__(
        self:    ZeroPad2d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies zero padding to 4-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        zeros. Only valid for 4-dimensional tensors (B, C, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 2 
                      integers, in which case it will be treated as 
                      (LR, TB), or a tuple of 4 integers for (L, R, T, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ZeroPad2d, x: Tensor) -> Tensor:
        assert x.ndim == 4, (
            'ZeroPad2d only accepts input Tensors with 4 dimensions '
            '([B, C, H, W]).')
        return pad(x, self.padding, 'constant', value=0)

class ZeroPad3d(Module):
    def __init__(
        self:    ZeroPad3d, 
        padding: builtins.int | tuple[builtins.int, ...]
    ) -> None:
        '''Applies zero padding to 5-dimensional input tensors.

        Pads input tensors by specified width, and fills padded area with
        zeros. Only valid for 5-dimensional tensors (B, C, D, H, W).
        
        Args:
            padding : The number of pixels to pad the input with. Can be a 
                      single integer, in which case the value will be used 
                      to, define the padding on all sides, or a tuple of 6 
                      integers for (L, R, T, B, F, B).
        '''
        super().__init__()
        self.padding = padding
        
    def forward(self: ZeroPad3d, x: Tensor) -> Tensor:
        assert x.ndim == 5, (
            'ZeroPad3d only accepts input Tensors with 5 dimensions '
            '([B, C, D, H, W]).')
        return pad(x, self.padding, 'constant', value=0)


