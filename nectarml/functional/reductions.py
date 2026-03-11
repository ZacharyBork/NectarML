from nectarml.tensor import Tensor
from nectarml.typing import ArrayLike

def min(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False
) -> Tensor:
    return input.min(dim, keepdim)

def max(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False
) -> Tensor:
    return input.max(dim, keepdim)

def argmin(
    input: Tensor,
    dim: int | None = None, 
    keepdim: bool = False
) -> ArrayLike:
    return input.argmin(dim, keepdim)
    
def argmax(
    input: Tensor,
    dim: int | None = None, 
    keepdim: bool = False
) -> ArrayLike:
    return input.argmax(dim, keepdim)

def mean(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False
) -> Tensor:
    return input.mean(dim, keepdim)

def sum(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    initial: int | float = 0
) -> Tensor:
    return input.sum(dim, keepdim, initial)

def prod(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    initial: int | float = 0
) -> Tensor:
    return input.prod(dim, keepdim, initial)
