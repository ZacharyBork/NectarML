
from nectarml.tensor import Tensor

### AVERAGE POOL ###

def avg_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True
) -> Tensor:
    stride = kernel_size if stride is None else stride
    pass

def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | float = None
) -> Tensor:
    stride = kernel_size if stride is None else stride
    pass

def avg_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | float = None
) -> Tensor:
    stride = kernel_size if stride is None else stride
    pass

### MAX POOL ###

def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor:
    stride = kernel_size if stride is None else stride
    pass

def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor:
    stride = kernel_size if stride is None else stride
    pass

def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor:
    stride = kernel_size if stride is None else stride
    pass

