from typing import Type, Literal
from dataclasses import dataclass

import numpy as np
from PIL import Image

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike

@dataclass
class TransformItem:
    datatype:   Type
    data:       Tensor | np.ndarray | Image.Image
    device:     Literal['cpu', 'cuda'] | None
    dtype:      DTypeLike | None
    normalized: bool
    max_value:  int | float | None


