from PIL import Image

from nectarml.vision.transforms import Transform

class Normalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class Denormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class MinMaxNormalize(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class ToFloat(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class ToUint8(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

