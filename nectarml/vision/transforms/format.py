from PIL import Image

from nectarml.vision.transforms import Transform

class ToTensor(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class ToPIL(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class ToNumpy(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class ConvertDtype(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class Permute(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

