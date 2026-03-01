from PIL import Image
import numpy as np

from nectarml import Tensor, DTypeLike, float32
import nectarml.functional as F

class Transform():
    def __init__(self) -> None:
        self.rng = np.random.default_rng()
    
    def _random_in_range(
        self, 
        value_range: tuple[int | float, int | float] = (0.0, 1.0)
    ) -> float:
        _min = value_range[0]
        _max = value_range[1]
        value = _min + (_max - _min) * self.rng.random() 
        return value
    
    def forward(self, input: Image.Image) -> Image.Image:
        raise NotImplementedError
    
    def __call__(self, input: Image.Image) -> Image.Image:
        return self.forward(input)
    

