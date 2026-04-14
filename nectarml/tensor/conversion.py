import builtins
from typing import Literal

import numpy as np

from .tensor import Tensor
from .bool import BoolTensor
from nectarml.cuda.utils import cast_tensor
from nectarml.typing import Size, bool_, uint8

def to_bool_tensor(
    input:  Tensor,
    data:   builtins.int | np.ndarray,
    shape:  Size | tuple[builtins.int, ...],
    device: Literal['cpu', 'cuda'] | None = None
) -> BoolTensor:
    '''Converts numerical Tensors to BoolTensors.
    
    Args:
        input : The numerical Tensor to convert.
        data : The data (either a uintptr to boolean tensor data in CUDA memory
            or a bool_ typed np.ndarray).
        shape : The shape of the new BoolTensor.
    
    Returns:
        BoolTensor : The newly created BoolTensor.
    '''
    device = input.device if device is None else device
    return BoolTensor._new(data, shape, bool_, device)

def to_numerical_tensor(input: BoolTensor) -> Tensor:
    '''Converts BoolTensor to numerical Tensor with dtype=uint8.
    
    Args:
        input : The BoolTensor to convert to numerical.
        
    Returns:
        Tensor : The newly created numerical Tensor.
    '''
    if input.device == 'cpu': data = input.data.astype(uint8)
    else: data = cast_tensor(input, new_dtype=uint8)
    return Tensor._new(data, input.shape, uint8, input.device)

