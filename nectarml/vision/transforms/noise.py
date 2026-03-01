from PIL import Image

from nectarml.vision.transforms import Transform

class GaussianNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class SaltAndPepperNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class SpeckleNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class ISONoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class MultiplicativeNoise(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

