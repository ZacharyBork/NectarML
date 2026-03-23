from typing import Any, Literal

from PIL import Image
import numpy as np

from nectarml.tensor import Tensor
from nectarml.vision import utils
from nectarml.typing import uint8

class Transform():
    def __init__(
        self, 
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        self.device = device
        self.original_type: Literal['image', 'ndarray', 'tensor'] = None
        self.rng = np.random.default_rng()
    
    ### UTILS ###
    
    def _random_in_range(
        self, 
        value_range: tuple[int | float, int | float] = (0.0, 1.0)
    ) -> float:
        _min = value_range[0]
        _max = value_range[1]
        value = _min + (_max - _min) * self.rng.random() 
        return value
    
    ### CONVERSION ###
    
    def _to_tensor(
        self, 
        input: Image.Image | np.ndarray | Tensor
    ) -> Tensor:
        if isinstance(input, Image.Image):
            self.original_type = 'image'
            input = utils.PIL_to_tensor(input, dtype=uint8,device='cpu')
        elif isinstance(input, np.ndarray):
            self.original_type = 'ndarray'
            input = Tensor(input.data, input.shape, input.dtype, device='cpu')
        elif isinstance(input, Tensor): self.original_type ='tensor'
        else: raise ValueError(
            f'Invalid input type for transform: {type(input)}')
        
        if self.device == 'auto': self.device = input.device
        else: input = input.to(self.device)
        
        return input
    
    def _from_tensor(
        self, 
        tensor: Tensor,
        to_type: Literal['pil', 'numpy'] | None = None
    ) -> Image.Image | np.ndarray | Tensor:
        to_type = to_type or self.original_type
        match to_type:
            case 'image': return Image.fromarray(tensor.numpy())
            case 'ndarray': return tensor.numpy()
            case 'tensor': return tensor
    
    ### FORWARD ###
    
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(
        self, 
        input: Image.Image | np.ndarray | Tensor
    ) -> Any:
        tensor = self._to_tensor(input)
        output = self.forward(tensor)
        return self._from_tensor(output)
    

