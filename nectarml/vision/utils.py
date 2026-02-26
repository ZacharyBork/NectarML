from os import PathLike
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

from PIL import Image
import numpy as np

from nectarml import Tensor, DTypeLike, float32, uint8, full
import nectarml.functional as F

### TENSOR UTILS ###

def _normalize_(
    input: Tensor, 
    range: tuple[int | float, int | float] = (255.0, 255.0)
) -> None:
    _min = input.min()
    _max = input.max()
    spatial = input[2:]
    spatial = ((spatial - _min) * ((range[1] - range[0]) / (_max - _min)))

def make_grid(
    input: Tensor | Sequence[Tensor], 
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: tuple[int, int] = [0, 255],
    scale_each: bool = False,
    pad_value: float = 0.0
) -> Tensor:
    if isinstance(input, Sequence): input = F.cat(input, dim=0)
    
    if scale_each:
        split = F.split(input, sizes=input.shape[0])
        if normalize: split = [_normalize_(i, value_range) for i in split]
    else:
        if normalize: input = _normalize_(input, value_range)
        split = F.split(input, sizes=input.shape[0])
                    
    count = len(split)
    rows, cols = int(np.ceil(count / nrow)), int(np.minimum(count, nrow))
    size = split[0].shape[-1] + (padding * 2)
    canvas = full((1, 3, size * rows, size * cols), fill_value=pad_value)
        
    curr_row = curr_col = 0
    for i in range(count):        
        start = (size * curr_row + padding, size * curr_col + padding)
        end = (size * (curr_row+1) - padding, size * (curr_col+1) - padding)
        canvas[:, :, start[0]:end[0], start[1]:end[1]] = split[i]
        
        if curr_col > cols - 2:
            curr_col = 0
            curr_row += 1
        else: curr_col += 1
    
    return canvas

### IMAGE I/O ###

def load_image(
    image_path: PathLike,
    dtype: DTypeLike = float32,
    device: Literal['cpu', 'cuda'] = 'cpu',
    normalize: bool = False,
    value_range: tuple[int | float, int | float] = [0.0, 1.0]
) -> Tensor: 
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f'Unable to locate image file at path: {image_path.as_posix()}')
    
    image = Image.open(image_path)
    data = np.array(image).astype(dtype)
    output = Tensor(data, dtype=dtype, device=device)
    
    output = output.permute((2, 0, 1))
    shape = (1,) + output.shape
    output = output.reshape(shape)
    
    if normalize: _normalize_(output, value_range)
    
    return output
    
def save_image(
    input: Tensor | Sequence[Tensor], 
    output_path: PathLike,
    normalize: bool = False,
    value_range: tuple[int, int] = [0, 255],
    **kwargs
) -> None: 
    output_path = Path(output_path)
    out_dir = output_path.parent.resolve()
    if not out_dir.exists():
        raise FileNotFoundError(
            f'Unable to locate output directory at path: {out_dir.as_posix()}')
        
    if isinstance(input, Sequence) or input.shape[0] > 1:
        output = make_grid(
            input, normalize=normalize, value_range=value_range, **kwargs)
    else:
        output = input
        if normalize: _normalize_(output, value_range)
            
    if output.ndim > 3: output = output.squeeze(dim=0)
    output = output.permute((1, 2, 0))
    image = Image.fromarray(output.data.astype(dtype=uint8), 'RGB')
    image.save(output_path)

