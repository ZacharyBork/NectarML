import builtins
import numpy  as np
from   typing import Any, TypeAlias

from . import return_types
from ._device import device
from ._size   import Size
from ._dtype  import (
    dtype, float, float16, float32, half, double,  int, int8, int16, int32, 
    int64, short, long, uint, uint8, uint16, uint32, uint64, bool_)

### COMMON TYPE ALIASING ###

DeviceLikeType: TypeAlias = builtins.str | device
DTypeLike:      TypeAlias = dtype | np.typing.DTypeLike
ArrayLike:      TypeAlias = np.typing.ArrayLike | list[Any] | tuple[Any, ...]

ShapeType: TypeAlias = (
    builtins.int
  | list [builtins.int]
  | tuple[builtins.int, ...]
  | Size
)

DimsType: TypeAlias = (
    builtins.int
  | list [builtins.int]
  | tuple[builtins.int, ...]
)

NumberType: TypeAlias = builtins.int | builtins.float 
