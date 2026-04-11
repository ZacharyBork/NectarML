from .tensor     import tensor
from .numerical  import Tensor
from .bool       import BoolTensor
from .conversion import to_bool_tensor, to_numerical_tensor

# Wiring for cross-class methods. Assigned here to avoid circular imports.
# Function definitions located at: nectarml/tensor/conversion.py
Tensor._to_bool          = to_bool_tensor
BoolTensor._to_numerical = to_numerical_tensor

