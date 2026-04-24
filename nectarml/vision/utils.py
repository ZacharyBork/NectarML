from os import PathLike
from pathlib import Path
from collections.abc import Sequence

from PIL import Image
import numpy as np

from nectarml.core   import Tensor
from nectarml          import typing
from nectarml.creation import full
import nectarml.functional as F

### TENSOR UTILS ###

def _normalize(
    input:       Tensor, 
    value_range: tuple[int | float, int | float] = (0.0, 255.0)
) -> Tensor:
    _min, _max = input.min().item(), input.max().item()
    _rmin, _rmax = value_range[0], value_range[1]
    if _min == _max:
        return input * 0.0 + _rmin + (_rmax - _rmin) * 0.5
    _range = _rmax - _rmin
    return (input - _min) * (_range / (_max - _min)) + _rmin

def make_grid(
    input:       Tensor | Sequence[Tensor], 
    nrow:        int = 8,
    padding:     int = 2,
    normalize:   bool = False,
    value_range: tuple[int, int] = (0, 255),
    scale_each:  bool = False,
    pad_value:   float = 0.0
) -> Tensor:
    if isinstance(input, Sequence): input = F.cat(input, dim=0)
    
    if scale_each:
        split = F.split(input, split_size=input.shape[0])
        if normalize: split = [_normalize(i, value_range) for i in split]
    else:
        if normalize: input = _normalize(input, value_range)
        split = F.split(input, split_size=input.shape[0])
                    
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

### CONVERSION ###

def tensor_to_PIL(
    input:       Tensor,
    normalize:   bool = False,
    value_range: tuple[int, int] = (0, 255)
) -> Image.Image:
    if input.ndim > 3: input = input.squeeze(dim=0)
    if input.shape[0] == 1: 
          mode, input = 'L',   input.squeeze(dim=0)
    else: mode, input = 'RGB', input.permute((1, 2, 0))
    if normalize: input = _normalize(input, value_range)
    return Image.fromarray(input.numpy().astype(dtype=np.uint8), mode)

### IMAGE I/O ###

def load_image(
    path:      PathLike,
    dtype:     typing.dtype = typing.float32,
    normalize: bool = False,
    batch_dim: bool = True
) -> Tensor: 
    '''Loads an image file as a nectarml.Tensor.
    
    Args:
        path      : The system path to the image file to load.
        dtype     : The dtype for the new tensor.
        normalize : Whether to normalize the output data. If True, the tensor
            will be divided by it's max item (plus a small epsilon value), 
            resulting in tensor with a saturated [0:1] range.
        batch_dim : Whether to add a batch dimension to the new tensor. If 
            True, the resulting tensor will have shape (B, C, H, W), if False,
            it will have shape (C, H, W).
            
    Returns:
        Tensor : The resulting image tensor.
    '''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Unable to locate image file at path: {path.as_posix()}')
        
    data   = np.array(Image.open(path)).astype(dtype.numpy)
    output = Tensor(data, dtype=dtype, device='cpu')
    output = output.permute((2, 0, 1)).contiguous()
    if batch_dim: output = output.unsqueeze(0)
    if normalize: output = output / (output.max().item() + 1e-8)
    return output
    
def save_image(
    input:       Tensor | Sequence[Tensor], 
    output_path: PathLike,
    normalize:   bool = False,
    value_range: tuple[int, int] = (0, 255),
    **kwargs
) -> None: 
    output_path = Path(output_path)
    out_dir     = output_path.parent.resolve()
    if not out_dir.exists():
        raise FileNotFoundError(
            f'Unable to locate output directory at path: {out_dir.as_posix()}')
    
    if isinstance(input, Sequence) or input.shape[0] > 1:
        input = make_grid(
            input, normalize=normalize, value_range=value_range, **kwargs)
        img = tensor_to_PIL(input, normalize=False)
    else: img = tensor_to_PIL(input, normalize, value_range)
    
    img.save(output_path)

