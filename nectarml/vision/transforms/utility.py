from nectarml.tensor import Tensor
from nectarml.vision.transforms.transform import Transform

class Permute(Transform[Tensor, Tensor]):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims
    
    def forward(self, input: Tensor) -> Tensor:
        return input.permute(self.dims)
    
class Transpose(Transform[Tensor, Tensor]):
    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, input: Tensor) -> Tensor:
        return input.transpose(self.dim1, self.dim2)
    
class Clamp(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        min_value: float | None = None,
        max_value: float | None = None
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, input: Tensor) -> Tensor:
        return input.clamp(self.min_value, self.max_value)
    
class MaskedFill(Transform[Tensor, Tensor]):
    def __init__(self, mask: Tensor, value: float = 0.0) -> None:
        super().__init__()
        self.mask = mask
        self.value = value
    
    def forward(self, input: Tensor) -> Tensor:
        return input.masked_fill(self.mask, self.value)

