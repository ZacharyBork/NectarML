import builtins
from typing      import Literal
from dataclasses import dataclass

import numpy as np

from .        import _tensor
from nectarml import typing
from nectarml.cuda.utils import cast_tensor

### PICKLING UTILS ###

def _reconstruct_tensor(
    data:          np.ndarray, 
    shape:         list[builtins.int], 
    dtype:         typing.dtype,
    device:        Literal['cpu', 'cuda'],
    requires_grad: builtins.bool
) -> _tensor.Tensor:
    return _tensor.Tensor(data, shape, dtype, device, requires_grad)

### FAKETENSOR UTILS ###

@dataclass(frozen=True)
class _to_fake:
    types = {
        None:            None,
        typing.float32:  _tensor.FloatTensor,
        typing.float16:  _tensor.HalfTensor,
        typing.int8:     _tensor.Int8Tensor,
        typing.uint8:    _tensor.Int8Tensor,
        typing.int16:    _tensor.Int16Tensor,
        typing.uint16:   _tensor.Int16Tensor,
        typing.int32:    _tensor.Int32Tensor,
        typing.uint32:   _tensor.Int32Tensor,
        typing.int64:    _tensor.Int64Tensor,
        typing.uint64:   _tensor.Int32Tensor
    }
    
    def __call__(
        self, 
        template: _tensor.tensor,
        dtype:    typing.dtype | None = None,
        device:   typing.DeviceLikeType | None = None,
        signed:   builtins.bool = True
    ) -> _tensor.fake.FakeTensor:
        return self.types[dtype](template, device, signed)

### BOOLTENSOR UTILS ###

def _to_bool_tensor(
    input:  _tensor.Tensor,
    data:   builtins.int | np.ndarray,
    shape:  typing.ShapeType,
    device: typing.DeviceLikeType | None = None
) -> _tensor.BoolTensor:
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
    return _tensor.BoolTensor._new(data, shape, typing.bool_, device)

def _to_numerical_tensor(input: _tensor.BoolTensor) -> _tensor.Tensor:
    '''Converts BoolTensor to numerical Tensor with dtype=uint8.
    
    Args:
        input : The BoolTensor to convert to numerical.
        
    Returns:
        Tensor : The newly created numerical Tensor.
    '''
    if input.device == 'cpu': data = input.data.astype(np.uint8)
    else: data = cast_tensor(input, new_dtype=typing.uint8)
    return _tensor.Tensor._new(data, input.shape, typing.uint8, input.device)


