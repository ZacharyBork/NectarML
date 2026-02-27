from typing import Literal

from nectarml import Tensor, DTypeLike, float32
import nectarml.nn as nn

class _Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(device, dtype)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        
    def forward(self, x: Tensor) -> Tensor:
        pass
    
class Conv1d(_Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        pass
    
class Conv2d(_Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        pass
    
class Conv3d(_Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        pass
    

