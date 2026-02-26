from PIL import Image

from nectarml.vision.transforms import Transform

class GaussianBlur(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class MotionBlur(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class MedianBlur(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class BoxBlur(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class RandomBlur(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class Sharpen(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class Emboss(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass

class UnsharpMask(Transform):
    def __init__(self) -> None:
        pass
    
    def run(self, input: Image.Image) -> Image.Image:
        pass
