import math
import random
from os import PathLike
from pathlib import Path
from contextlib import nullcontext

from nectarml.tensor import Tensor
from nectarml.vision.transforms import Transform, format, utility
from nectarml.benchmark import benchmark_time

class Compose(Transform):
    def __init__(
        self, 
        transforms: list[Transform]
    ) -> None:
        super().__init__()
        self.transforms = transforms
    
    def forward(self, input: Tensor) -> Tensor:
        for transform in self.transforms:
            input = transform.forward(input)
        return input
    
    def optimize(self) -> None:
        optimized = []
        f = format
        for idx, xform in enumerate(self.transforms):
            if idx == len(self.transforms): break
            
            datatypes = (f.ToTensor, f.ToPIL, f.ToNumpy)
            if isinstance(xform, datatypes):
                next_mod = self.transforms[idx+1]
                if isinstance(next_mod, datatypes): continue
            
            casts = (f.ToCPU, f.ToCUDA, f.ChangeDevice)
            if isinstance(xform, casts):
                next_mod = self.transforms[idx+1]
                if isinstance(next_mod, casts): continue
            
            optimized.append(xform)
        self.transforms = optimized
     
    def _generate_examples(
        self,
        input_image: PathLike,
        output_directory: PathLike,
        num_examples: int,
        allow_overwrite: bool,
        benchmark: bool,
        make_grid: bool,
        **grid_kwargs
    ) -> list[Tensor]:
        input_image = utility.LoadImageFile(input_image)()
        outputs = []
        
        for i in range(num_examples):
            iter_context = benchmark_time(f'Iteration {i+1}') \
                if benchmark else nullcontext()

            with iter_context:
                output = self.forward(input_image)
                output_path = Path(output_directory, f'example_{i+1}.jpg')
                
                if not allow_overwrite:
                    assert not output_path.exists(), (
                        f'Found existing file at path: '
                        f'{output_path.as_posix()}\n'
                        f'Remove existing file or run generate_examples '
                        f'with allow_overwrite=True to continue.')
            
            if not make_grid: utility.SaveImageFile(output_path)(output)
            else: outputs.append(output)
        
        if len(outputs) > 0:
            grid = utility.MakeGrid(**grid_kwargs)(outputs)
            output_path = Path(output_directory, f'example_grid.jpg')
            utility.SaveImageFile(output_path)(grid)
        
    def generate_examples(
        self, 
        input_image: PathLike,
        output_directory: PathLike,
        num_examples: int = 5,
        allow_overwrite: bool = False,
        benchmark: bool = False,
        make_grid: bool = True,
        **grid_kwargs
    ) -> None:
        input_image = Path(input_image).resolve()
        assert input_image.exists(), \
            f'Unable to locate image file at path: {input_image.as_posix()}'
            
        output_directory = Path(output_directory).resolve()
        assert output_directory.exists(), (
            f'Unable to locate output directory at path: '
            f'{output_directory.as_posix()}')
        
        for xform in self.transforms:
            if isinstance(xform, utility.LoadImageFile):
                raise RuntimeError(
                    'Unable to run example generation on Compose which '
                    'contains LoadImageFile Transform.')
            if isinstance(xform, utility.SaveImageFile):
                raise RuntimeError(
                    'Unable to run example generation on Compose which '
                    'contains SaveImageFile Transform.')
        
        global_context = benchmark_time('Full Test') \
            if benchmark else nullcontext()
             
        with global_context:
            self._generate_examples(
                input_image, output_directory, num_examples,allow_overwrite,
                benchmark, make_grid, **grid_kwargs)
        
    def __call__(self, input: Tensor | None = None) -> Tensor:
        return self.forward(input)
    
    def __repr__(self) -> str:
        output = '\nCompose:\n\n'
        for idx, xform in enumerate(self.transforms):
            output += f'    {idx} : {xform}\n'
        return output

class RandomApply(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        transforms: list[Transform],
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.probability = p
    
    def forward(self, input: Tensor) -> Tensor:
        threshold = self._random_in_range()
        if threshold <= self.probability:
            for transform in self.transforms:
                input = transform.forward(input)
        return input

class RandomChoice(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        transforms: list[Transform],
        p: float | list[float] = 0.5
    ) -> None:
        self.use_prob_list = isinstance(p, list)
        if self.use_prob_list:
            assert len(p) == len(transforms), (
                'Probabilities list must be of equal length to xforms list.')
        else: p = [p] * len(transforms)
        
        super().__init__()
        self.transforms = transforms
        self.probabilities = p
    
    def forward(self, input: Tensor) -> Tensor:
        for xform, p in list(zip(self.transforms, self.probabilities)):
            threshold = self._random_in_range()
            if threshold <= p: input = xform.forward(input)
        return input

class RandomOrder(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        transforms: list[Transform]
    ) -> None:
        super().__init__()
        self.transforms = transforms
    
    def forward(self, input: Tensor) -> Tensor:
        xforms = self.transforms.copy()
        random.shuffle(xforms)
        for transform in xforms: input = transform.forward(input)
        return input

class OneOf(Transform[Tensor, Tensor]):
    def __init__(
        self, 
        transforms: list[Transform]
    ) -> None:
        super().__init__()
        self.transforms = transforms
    
    def forward(self, input: Tensor) -> Tensor:
        rand = self._random_in_range((0, len(self.transforms)))
        xform = self.transforms[int(math.floor(rand))]
        return xform.forward(input)


