from collections.abc import Sequence

from nectarml import Tensor

# NORMALIZATION

def BatchNorm1d(
    x: Tensor,
    num_features: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def BatchNorm2d(
    x: Tensor,
    num_features: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def BatchNorm3d(
    x: Tensor,
    num_features: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def InstanceNorm1d(
    x: Tensor,
    num_features: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def InstanceNorm2d(
    x: Tensor,
    num_features: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def InstanceNorm3d(
    x: Tensor,
    num_features: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def GroupNorm(
    x: Tensor,
    num_groups: int,
    num_channels: int,
    eps: float = 0.00001
) -> Tensor:
    pass

def LayerNorm(
    x: Tensor,
    normalized_shape: Sequence[int],
    eps: float = 0.0001
) -> Tensor:
    pass

