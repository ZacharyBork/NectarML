from __future__ import annotations

import math
from os              import PathLike
from typing          import Any
from pathlib         import Path
from contextlib      import nullcontext
from collections.abc import Iterator, Iterable

from nectarml.core                        import Tensor
from nectarml.vision.transforms.transform import Transform
from nectarml.vision.transforms.common    import TransformInput
from nectarml.vision.transforms           import format, utility
from nectarml.utils.benchmark             import benchmark_time

class Compose(Transform):
    def __init__(
        self, 
        *transforms: Transform | Iterable[Transform]
    ) -> None:
        super().__init__()
        
        if len(transforms) == 1:
            if isinstance(transforms[0], Transform): transforms = [transforms]
            if isinstance(transforms[0], list | tuple):
                transforms = list(transforms[0])
        else: transforms = list(transforms)
        
        self.transforms = transforms
    
    ### List-like methods ###
    
    def append(self, transform: Transform) -> None:
        self.transforms.append(transform)
    
    def extend(self, other: Compose | list[Transform]) -> None:
        if isinstance(other, Compose): other = other.transforms
        self.transforms.extend(other)
    
    def insert(self, index: int, transform: Transform) -> None:
        self.transforms.insert(index, transform)  
    
    def __getitem__(self, index: int) -> Transform:
        return self.transforms[index]

    def __setitem__(self, index: int, transform: Transform) -> None:
        self.transforms[index] = transform

    def __delitem__(self, index: int) -> None:
        del self.transforms[index]

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> Iterator[Transform]:
        return iter(self.transforms)

    def __contains__(self, transform: Transform) -> bool:
        return transform in self.transforms
    
    ### Forward method ###
    
    def forward(self, input: TransformInput) -> TransformInput:
        for transform in self.transforms: input = transform._call(input)
        return input
    
    ### Optimization ###
    
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
     
    ### Example generation ###
     
    def _generate_examples(
        self,
        input_image:  PathLike,
        num_examples: int,
        benchmark:    bool
    ) -> list[Tensor]:
        input_image = utility.LoadImageFile(input_image)()
        outputs = []
        
        for i in range(num_examples):
            iter_context = benchmark_time(f'Iteration {i+1}') \
                if benchmark else nullcontext()

            with iter_context:
                output = self.forward(input_image)
                
            
            outputs.append(output)
        return outputs
        
    def generate_examples(
        self, 
        input_image:      PathLike,
        output_directory: PathLike,
        num_examples:     int  = 5,
        allow_overwrite:  bool = False,
        benchmark:        bool = False,
        make_grid:        bool = True,
        **grid_kwargs:    dict[str, Any]
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
            outputs = self._generate_examples(
                input_image, num_examples, benchmark)
            
        if make_grid:
            grid = utility.MakeGrid(**grid_kwargs)(outputs)
            output_path = Path(output_directory, f'example_grid.jpg')
            utility.SaveImageFile(output_path)(grid)
        else:
            for i, output in enumerate(outputs):
                output_path = Path(output_directory, f'example_{i+1}.jpg')
                if not allow_overwrite:
                    assert not output_path.exists(), (
                        f'Found existing file at path: '
                        f'{output_path.as_posix()}\n'
                        f'Remove existing file or run generate_examples '
                        f'with allow_overwrite=True to continue.')
                utility.SaveImageFile(output_path)(output)
    
    ### Inspection ###
    
    def __repr__(self) -> str:
        output = '\nCompose:\n\n'
        for idx, xform in enumerate(self.transforms):
            output += f'    {idx} : {xform}\n'
        return output

class RandomApply(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        p:          float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.transforms = transforms
    
    def forward(self, input: TransformInput) -> TransformInput:
        threshold = self._random_in_range()
        if threshold <= self.p:
            for transform in self.transforms:
                input = transform._call(input)
        return input

class RandomChoice(Transform):
    def __init__(
        self, 
        transforms: list[Transform],
        p:          float | list[float] = 0.5
    ) -> None:
        self.use_prob_list = isinstance(p, list)
        if self.use_prob_list:
            assert len(p) == len(transforms), (
                'Probabilities list must be of equal length to xforms list.')
        else: p = [p] * len(transforms)
        
        super().__init__()
        self.transforms = transforms
        self.probabilities = p
    
    def forward(self, input: TransformInput) -> TransformInput:
        for xform, p in list(zip(self.transforms, self.probabilities)):
            threshold = self._random_in_range()
            if threshold <= p: input = xform._call(input)
        return input

class RandomOrder(Transform):
    def __init__(
        self, 
        transforms: list[Transform]
    ) -> None:
        super().__init__()
        self.transforms = transforms
    
    def forward(self, input: TransformInput) -> TransformInput:
        xforms = self.transforms.copy()
        self.rng.shuffle(xforms)
        for transform in xforms: input = transform._call(input)
        return input

class OneOf(Transform):
    def __init__(
        self, 
        transforms: list[Transform]
    ) -> None:
        super().__init__()
        self.transforms = transforms
    
    def forward(self, input: TransformInput) -> TransformInput:
        rand = self._random_in_range((0, len(self.transforms)))
        xform = self.transforms[int(math.floor(rand))]
        return xform._call(input)


