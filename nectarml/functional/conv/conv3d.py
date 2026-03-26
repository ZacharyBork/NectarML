from typing import Literal

from nectarml.tensor import Tensor
from nectarml import cpu, cuda

def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> Tensor:
    raise NotImplementedError('3D convolution is currently not supported.')

