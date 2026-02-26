from PIL import Image
import numpy as np

from nectarml import Tensor, DTypeLike, float32
import nectarml.functional as F

class Transform():
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        raise NotImplementedError
    
    def __call__(self, input: Image.Image) -> Image.Image:
        return self.run(input)
    

