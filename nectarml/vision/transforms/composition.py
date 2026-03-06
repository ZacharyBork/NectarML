import math
import random
from typing import Literal

from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform

class Compose(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        super().__init__(device)
        self.transforms = transforms
    
    def forward(self, input: Tensor) -> Tensor:
        for transform in self.transforms:
            input = transform.forward(input)
        return input

class RandomApply(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        p: float = 0.5,
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        super().__init__(device)
        self.transforms = transforms
        self.probability = p
    
    def forward(self, input: Tensor) -> Tensor:
        threshold = self._random_in_range()
        if threshold <= self.probability:
            for transform in self.transforms:
                input = transform.forward(input)
        return input

class RandomChoice(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        p: float | list[float] = 0.5,
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        self.use_prob_list = isinstance(p, list)
        if self.use_prob_list:
            assert len(p) == len(transforms), (
                'Probabilities list must be of equal length to xforms list.')
        else: p = [p] * len(transforms)
        
        super().__init__(device)
        self.transforms = transforms
        self.probabilities = p
    
    def forward(self, input: Tensor) -> Tensor:
        for xform, p in list(zip(self.transforms, self.probabilities)):
            threshold = self._random_in_range()
            if threshold <= p: input = xform.forward(input)
        return input

class RandomOrder(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        super().__init__(device)
        self.transforms = transforms
    
    def forward(self, input: Tensor) -> Tensor:
        xforms = self.transforms.copy()
        random.shuffle(xforms)
        for transform in xforms: input = transform.forward(input)
        return input

class OneOf(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ) -> None:
        super().__init__(device)
        self.transforms = transforms
    
    def forward(self, input: Tensor) -> Tensor:
        rand = self._random_in_range((0, len(self.transforms)))
        xform = self.transforms[int(math.floor(rand))]
        return xform.forward(input)


