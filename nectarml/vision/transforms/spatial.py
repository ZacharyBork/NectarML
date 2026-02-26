from PIL import Image

from nectarml.vision.transforms import Transform

class RandomCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass
    
class CenterCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass
    
class RandomResizedCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass
    
class Resize(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomHorizontalFlip(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomVerticalFlip(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomRotation(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomAffine(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomPerspective(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class ElasticTransform(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class GridDistortion(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class OpticalDistortion(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class Pad(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomPad(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class FiveCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class TenCrop(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomCropNearBBox(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

