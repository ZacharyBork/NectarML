from __future__ import annotations

from nectarml.core       import Tensor
from nectarml.nn.module  import Module
from nectarml.functional import pooling

### AVERAGE POOL ###

class AvgPool1d(Module):
    def __init__(
        self:              AvgPool1d, 
        kernel_size:       int | tuple[int],
        stride:            int | tuple[int] | None = None,
        padding:           int | tuple[int] = 0,
        ceil_mode:         bool = False,
        count_include_pad: bool = True
    ) -> None:
        super().__init__()
        self.kernel_size       = kernel_size
        self.stride            = stride
        self.padding           = padding
        self.ceil_mode         = ceil_mode
        self.count_include_pad = count_include_pad
        
    def forward(self: AvgPool1d, x: Tensor) -> Tensor:
        return pooling.avg_pool1d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
            self.count_include_pad)
        
class AvgPool2d(Module):
    def __init__(
        self:              AvgPool2d, 
        kernel_size:       int | tuple[int, int],
        stride:            int | tuple[int, int] | None = None,
        padding:           int | tuple[int, int] = 0,
        ceil_mode:         bool = False,
        count_include_pad: bool = True,
        divisor_override:  int | float | None = None
    ) -> None:
        super().__init__()
        self.kernel_size       = kernel_size
        self.stride            = stride
        self.padding           = padding
        self.ceil_mode         = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override  = divisor_override

    def forward(self: AvgPool2d, x: Tensor) -> Tensor:
        return pooling.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
            self.count_include_pad, self.divisor_override)
    
class AvgPool3d(Module):
    def __init__(
        self:              AvgPool3d, 
        kernel_size:       int | tuple[int, int, int],
        stride:            int | tuple[int, int, int] | None = None,
        padding:           int | tuple[int, int, int] = 0,
        ceil_mode:         bool = False,
        count_include_pad: bool = True,
        divisor_override:  int | float | None = None
    ) -> None:
        super().__init__()
        self.kernel_size       = kernel_size
        self.stride            = stride
        self.padding           = padding
        self.ceil_mode         = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override  = divisor_override
    
    def forward(self: AvgPool3d, x: Tensor) -> Tensor:
        return pooling.avg_pool3d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
            self.count_include_pad, self.divisor_override)

### MAX POOL ###

class MaxPool1d(Module):
    def __init__(
        self:           MaxPool1d, 
        kernel_size:    int | tuple[int],
        stride:         int | tuple[int] | None = None,
        padding:        int | tuple[int] = 0,
        dilation:       int = 1,
        ceil_mode:      bool = False,
        return_indices: bool = False
    ) -> None:
        super().__init__()
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.ceil_mode      = ceil_mode
        self.return_indices = return_indices
            
    def forward(self: MaxPool1d, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return pooling.max_pool1d(
            x, self.kernel_size, self.stride, self.padding, self.dilation, 
            self.ceil_mode, self.return_indices)

class MaxPool2d(Module):
    def __init__(
        self:           MaxPool2d, 
        kernel_size:    int | tuple[int, int],
        stride:         int | tuple[int, int] | None = None,
        padding:        int | tuple[int, int] = 0,
        dilation:       int = 1,
        ceil_mode:      bool = False,
        return_indices: bool = False
    ) -> None:
        super().__init__()
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.ceil_mode      = ceil_mode
        self.return_indices = return_indices
    
    def forward(self: MaxPool2d, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return pooling.max_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation, 
            self.ceil_mode, self.return_indices)
    
class MaxPool3d(Module):
    def __init__(
        self:           MaxPool3d, 
        kernel_size:    int | tuple[int, int, int],
        stride:         int | tuple[int, int, int] | None = None,
        padding:        int | tuple[int, int, int] = 0,
        dilation:       int = 1,
        ceil_mode:      bool = False,
        return_indices: bool = False
    ) -> None:
        super().__init__()
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.ceil_mode      = ceil_mode
        self.return_indices = return_indices

    def forward(self: MaxPool3d, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return pooling.max_pool3d(
            x, self.kernel_size, self.stride, self.padding, self.dilation, 
            self.ceil_mode, self.return_indices)

