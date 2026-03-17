from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import _nectarml
from nectarml.cuda.utils import map_dtype

### 1-Dimensional ###

def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    B: int, C_in: int, L: int,
    C_out: int, K: int,
    stride: int, 
    padding: int, 
    dilation: int,
    groups: int
) -> int:
    assert groups == 1, \
        'Grouped convolution is not currently supported for CUDA Tensors.'

    return _nectarml.conv1d(
        input._data_ptr, weight._data_ptr,
        bias._data_ptr if bias is not None else 0,
        B, C_in, L, C_out, K,
        stride, padding, dilation, groups,
        map_dtype(input.dtype))

def conv1d_backward_input(
    out_grad: Tensor,
    weight: Tensor,
    B: int, C_in: int, L: int,
    C_out: int, K: int, L_out: int,
    stride: int, padding: int, dilation: int
) -> int:
    return _nectarml.conv1d_backward_input(
        out_grad._data_ptr, weight._data_ptr,
        B, C_in, L, C_out, K, L_out,
        stride, padding, dilation,
        map_dtype(out_grad.dtype))
    
def conv1d_backward_weight(
    out_grad: Tensor,
    input: Tensor,
    B: int, C_in: int, L: int,
    C_out: int, K: int, L_out: int,
    stride: int, padding: int, dilation: int
) -> int:
    return _nectarml.conv1d_backward_weight(
        out_grad._data_ptr, input._data_ptr,
        B, C_in, L, C_out, K, L_out,
        stride, padding, dilation,
        map_dtype(out_grad.dtype))
    
### 2-Dimensional ###



### 3-Dimensional ###

