import builtins

import numpy as np

from .tensor import Tensor
from .bool import BoolTensor
from nectarml.cuda.utils import cast_tensor
from nectarml import typing

def to_bool_tensor(
    input:  Tensor,
    data:   builtins.int | np.ndarray,
    shape:  typing.ShapeType,
    device: typing.DeviceLikeType | None = None
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
    return BoolTensor._new(data, shape, typing.bool_, device)

def to_numerical_tensor(input: BoolTensor) -> Tensor:
    '''Converts BoolTensor to numerical Tensor with dtype=uint8.
    
    Args:
        input : The BoolTensor to convert to numerical.
        
    Returns:
        Tensor : The newly created numerical Tensor.
    '''
    if input.device == 'cpu': data = input.data.astype(np.uint8)
    else: data = cast_tensor(input, new_dtype=typing.uint8)
    return Tensor._new(data, input.shape, typing.uint8, input.device)

