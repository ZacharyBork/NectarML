from PIL import Image

from nectarml.vision.transforms import Transform

class Compose(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomApply(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomChoice(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class RandomOrder(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class OneOf(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

class SomeOf(Transform):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Image.Image) -> Image.Image:
        pass

