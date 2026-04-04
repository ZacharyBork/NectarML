from typing import Literal

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import Size
from nectarml.creation import linspace

### UTILS ###

def apply_kernel_2d(image: Tensor, kernel: Tensor) -> Tensor:
    B, C, H, W = image.shape
    KH, KW = kernel.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    image_flat = image.reshape((B * C, 1, H, W))
    result = F.conv2d(image_flat, kernel, padding=(KH//2, KW//2), groups=1)
    return result.reshape((B, C, H, W))

def lerp(a: Tensor, b: Tensor, w: Tensor) -> Tensor:
    return a + w * (b - a)

def lerp3(a: Tensor, b: Tensor, c: Tensor, w: Tensor) -> Tensor:
    w1 = (w * 2).clamp(0.0, 1.0)
    w2 = ((w - 0.5) * 2).clamp(0.0, 1.0)
    t  = (w >= 0.5).to(w.device, w.dtype)
    return lerp(lerp(a, b, w1), lerp(b, c, w2), t)

def gradient_mask(
    shape: tuple[int, ...] | Size,
    mode: Literal[
        'horizontal', 'vertical', 'radial', 'elliptical'
    ] = 'horizontal',
    radial_method: Literal['corners', 'edges'] = 'corners',
    invert: bool = False
) -> Tensor:
    _, _, H, W = shape
    match mode:
        case 'horizontal': 
            mask = linspace(0, 1, W).reshape((1, W)).expand((H, W))
            mask = mask.reshape((1, 1, H, W))
        case 'vertical':
            mask = linspace(0, 1, H).reshape((H, 1)).expand((H, W))
            mask = mask.reshape((1, 1, H, W))
        case 'radial': 
            yy = linspace(-1, 1, H).reshape((H, 1)).expand((H, W))
            xx = linspace(-1, 1, W).reshape((1, W)).expand((H, W))
            if radial_method == 'corners':
                dist = (xx**2 + yy**2).sqrt()
                mask = (dist / dist.max()).reshape((1, 1, H, W))
            else:
                dist = (xx**2 + yy**2).sqrt()
                mask = dist.clamp(0.0, 1.0).reshape((1, 1, H, W))
        case 'elliptical':
            yy = linspace(-1, 1, H).reshape((H, 1)).expand((H, W))
            xx = linspace(-1, 1, W).reshape((1, W)).expand((H, W))
            dist = (xx**2 + yy**2).sqrt() / (2**0.5)
            mask = dist.clamp(0.0, 1.0).reshape((1, 1, H, W))
    
    mask = mask.clamp(0.0, 1.0)
    if invert: mask = 1 - mask
    return mask
        
