from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

class GaussianBlur(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class MotionBlur(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class MedianBlur(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class BoxBlur(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomBlur(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Sharpen(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class Emboss(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class UnsharpMask(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass
