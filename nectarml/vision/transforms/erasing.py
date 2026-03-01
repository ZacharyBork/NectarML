from PIL import Image

from nectarml.vision.transforms import Transform

class RandomErasing(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class CoarseDropout(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class GridDropout(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomSunFlare(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomFog(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomRain(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomSnow(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomShadow(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

