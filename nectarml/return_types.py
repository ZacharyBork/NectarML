from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml.core import Tensor

from dataclasses import dataclass

### REDUCTION RETURN TYPES ###

@dataclass(frozen=True)
class min:
    values:  Tensor
    indices: Tensor

@dataclass(frozen=True)
class max:
    values:  Tensor
    indices: Tensor



