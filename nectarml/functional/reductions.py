from nectarml import Tensor, ArrayLike
from nectarml.functional.common import _eval_core_function
from nectarml._core import reductions

def min(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> Tensor:
    return _eval_core_function(
        input, lambda x : reductions.min(x, dim=dim, keepdims=keepdims))

def max(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> Tensor:
    return _eval_core_function(
        input, lambda x : reductions.max(x, dim=dim, keepdims=keepdims))

def argmin(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return reductions.argmin(input.data, dim=dim, keepdims=keepdims)
    
def argmax(
    input: Tensor,
    dim: int | None = None, 
    keepdims: bool = False
) -> ArrayLike:
    return reductions.argmax(input.data, dim=dim, keepdims=keepdims)

def mean(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> Tensor:
    return _eval_core_function(
        input, lambda x : reductions.mean(x, dim=dim, keepdims=keepdims))

def sum(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> Tensor:
    return _eval_core_function(
        input, lambda x : reductions.sum(x, dim, keepdims, initial))

def prod(
    input: Tensor, 
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    initial: int | float = 0
) -> Tensor:
    return _eval_core_function(
        input, lambda x : reductions.prod(x, dim, keepdims, initial))
