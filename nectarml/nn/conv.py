from __future__ import annotations

import math
from typing import Literal

import nectarml.functional as F
from nectarml              import typing
from nectarml.tensor       import Tensor
from nectarml.nn.module    import Module
from nectarml.creation     import empty
from nectarml.nn.init      import kaiming_uniform_, uniform_
from nectarml.amp.autocast import autocast_state

### 1-Dimensional ###
    
class Conv1d(Module):
    def __init__(
        self: Conv1d,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int = 1,
        padding:      int = 0,
        dilation:     int = 1,
        groups:       int = 1,
        bias:        bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        dtype: typing.dtype = typing.float32
    ) -> None:
        super().__init__(dtype)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups
        self.padding_mode = padding_mode

        self.weight = empty(
            (out_channels, in_channels // groups, kernel_size),
            dtype=self.dtype, device='cpu', requires_grad=True)
        kaiming_uniform_(
            self.weight, mode='fan_in', 
            nonlinearity='leaky_relu', a=math.sqrt(5))

        if bias:
            k = (in_channels // groups) * kernel_size
            bound = 1.0 / k ** 0.5
            
            self.bias = empty(
                (out_channels,), dtype=self.dtype, 
                device='cpu', requires_grad=True)
            uniform_(self.bias, -bound, bound)
        else: self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        state       = autocast_state()
        input_dtype = x.dtype
        if state.enabled and state.context == 'cuda':
            x      = x.to(dtype=typing.float16)
            weight = self.weight.to(dtype=typing.float16)
            bias   = self.bias.to(dtype=typing.float16) \
                  if self.bias is not None else None
        else:
            weight = self.weight
            bias   = self.bias
        out = F.conv1d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode)
        return out.to(dtype=input_dtype)

class ConvTranspose1d(Module):
    def __init__(
        self: ConvTranspose1d,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int,
        stride:         int = 1,
        padding:        int = 0,
        output_padding: int = 0,
        dilation:       int = 1,
        groups:         int = 1,
        bias:          bool = True,
        dtype: typing.dtype = typing.float32
    ) -> None:
        super().__init__(dtype)
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.output_padding = output_padding
        self.dilation       = dilation
        self.groups         = groups

        self.weight = empty(
            (in_channels, out_channels // groups, kernel_size),
            dtype=self.dtype, device='cpu', requires_grad=True)
        kaiming_uniform_(
            self.weight, mode='fan_in', 
            nonlinearity='leaky_relu', a=math.sqrt(5))

        if bias:
            k = (in_channels // groups) * kernel_size
            bound = 1.0 / k ** 0.5
            
            self.bias = empty(
                (out_channels,), dtype=self.dtype, 
                device='cpu', requires_grad=True)
            uniform_(self.bias, -bound, bound)
        else: self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        state       = autocast_state()
        input_dtype = x.dtype
        if state.enabled and state.context == 'cuda':
            x      = x.to(dtype=typing.float16)
            weight = self.weight.to(dtype=typing.float16)
            bias   = self.bias.to(dtype=typing.float16) \
                  if self.bias is not None else None
        else:
            weight = self.weight
            bias   = self.bias
        out = F.conv_transpose1d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode='zeros')
        return out.to(dtype=input_dtype)
    
### 2-Dimensional ###
    
class Conv2d(Module):
    def __init__(
        self: Conv2d,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int | tuple[int, int],
        stride:       int | tuple[int, int] = 1,
        padding:      int | tuple[int, int] = 0,
        dilation:     int | tuple[int, int] = 1,
        groups:       int = 1,
        bias:        bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        dtype: typing.dtype = typing.float32
    ) -> None:
        super().__init__(dtype)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups
        self.padding_mode = padding_mode

        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size

        self.weight = empty(
            (out_channels, in_channels // groups) + self.kernel_size,
            dtype=self.dtype, device='cpu', requires_grad=True)
        kaiming_uniform_(
            self.weight, mode='fan_in', 
            nonlinearity='leaky_relu', a=math.sqrt(5))

        if bias:
            k = (in_channels // groups) \
              * self.kernel_size[0] \
              * self.kernel_size[1]
            bound = 1.0 / k ** 0.5
            self.bias = empty(
                (out_channels,), dtype=self.dtype,
                device='cpu', requires_grad=True)
            uniform_(self.bias, -bound, bound)
        else: self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        state       = autocast_state()
        input_dtype = x.dtype
        if state.enabled and state.context == 'cuda':
            x      = x.to(dtype=typing.float16)
            weight = self.weight.to(dtype=typing.float16)
            bias   = self.bias.to(dtype=typing.float16) \
                  if self.bias is not None else None
        else:
            weight = self.weight
            bias   = self.bias
        out = F.conv2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode)
        return out.to(dtype=input_dtype)

    
class ConvTranspose2d(Module):
    def __init__(
        self: ConvTranspose2d,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int | tuple[int, int],
        stride:         int | tuple[int, int] = 1,
        padding:        int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        dilation:       int | tuple[int, int] = 1,
        groups:         int = 1,
        bias:          bool = True,
        dtype: typing.dtype = typing.float32
    ) -> None:
        super().__init__(dtype)
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.stride         = stride
        self.padding        = padding
        self.output_padding = output_padding
        self.dilation       = dilation
        self.groups         = groups

        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size

        self.weight = empty(
            (in_channels, out_channels // groups) + self.kernel_size,
            dtype=self.dtype, device='cpu', requires_grad=True)
        kaiming_uniform_(
            self.weight, mode='fan_in', 
            nonlinearity='leaky_relu', a=math.sqrt(5))

        if bias:
            k = (in_channels // groups) \
              * self.kernel_size[0] \
              * self.kernel_size[1]
            bound = 1.0 / k ** 0.5
            self.bias = empty(
                (out_channels,), dtype=self.dtype,
                device='cpu', requires_grad=True)
            uniform_(self.bias, -bound, bound)
        else: self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        state       = autocast_state()
        input_dtype = x.dtype
        if state.enabled and state.context == 'cuda':
            x      = x.to(dtype=typing.float16)
            weight = self.weight.to(dtype=typing.float16)
            bias   = self.bias.to(dtype=typing.float16) \
                  if self.bias is not None else None
        else:
            weight = self.weight
            bias   = self.bias
        out = F.conv_transpose2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode='zeros')
        return out.to(dtype=input_dtype)

### 3-Dimensional ###
    
class Conv3d(Module):
    def __init__(
        self: Conv3d,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int | tuple[int, int, int],
        stride:       int | tuple[int, int, int] = 1,
        padding:      int | tuple[int, int, int] = 0,
        dilation:     int | tuple[int, int, int] = 1,
        groups:       int = 1,
        bias:        bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        dtype: typing.dtype = typing.float32
    ) -> None:
        raise NotImplementedError('3D convolution is not currently supported.')
        super().__init__(dtype)

    def forward(self: Conv3d, x: Tensor) -> Tensor:        
        raise NotImplementedError
    
class ConvTranspose3d(Module):
    def __init__(
        self: ConvTranspose3d,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int | tuple[int, int, int],
        stride:         int | tuple[int, int, int] = 1,
        padding:        int | tuple[int, int, int] = 0,
        output_padding: int | tuple[int, int, int] = 0,
        dilation:       int | tuple[int, int, int] = 1,
        groups:         int = 1,
        bias:          bool = True,
        padding_mode: Literal[
            'zeros', 'reflect', 'replicate', 'circular'
        ] = 'zeros',
        dtype: typing.dtype = typing.float32
    ) -> None:
        raise NotImplementedError('3D convolution is not currently supported.')
        super().__init__(dtype)
        

    def forward(self: ConvTranspose3d, x: Tensor) -> Tensor:        
        raise NotImplementedError

