import numpy as np

from nectarml import Tensor

def softmax(x: Tensor) -> Tensor:
    exp_x = (x - x.max(dim=-1, keepdims=True)).exp()
    return exp_x / exp_x.sum(dim=-1, keepdims=True)

