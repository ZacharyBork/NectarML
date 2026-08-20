from . import _tensor
from ._tensor import (
    tensor, 
    Tensor,
    BoolTensor,
)
from .utils import (
    _to_fake, 
    _to_bool_tensor, 
    _to_numerical_tensor,
    _reconstruct_tensor
)

T = _tensor.tensor
T._fake                     = _to_fake()
T._reconstruct              = _reconstruct_tensor
T._subclasses['Tensor']     = Tensor,
T._subclasses['BoolTensor'] = BoolTensor

# Wiring for cross-class methods. Assigned here to avoid circular imports.
# Function definitions located at: nectarml/tensor/conversion.py
Tensor._to_bool          = _to_bool_tensor
BoolTensor._to_numerical = _to_numerical_tensor

