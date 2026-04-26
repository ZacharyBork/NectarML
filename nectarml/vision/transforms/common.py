from __future__ import annotations

import colorsys
from warnings    import warn
from typing      import Self, Any, Literal
from dataclasses import dataclass

import numpy as np
from PIL import Image

import _nectarml
import nectarml.functional as F
from nectarml.core     import Tensor
from nectarml.typing   import Size
from nectarml.creation import linspace

### DATA ###

@dataclass
class TransformInput:
    image:     Tensor | np.ndarray | Image.Image
    image2:    Tensor | np.ndarray | Image.Image | None = None
    mask:      Tensor | np.ndarray | Image.Image | None = None
    boxes:     Tensor | np.ndarray | Image.Image | None = None
    keypoints: Tensor | np.ndarray | Image.Image | None = None

    def _istensor(self: TransformInput, value: Any) -> bool:
        return value is not None and isinstance(value, Tensor)

    def _all_inputs(self: TransformInput) -> list[Any]:
        return [
            self.image, 
            self.image2, 
            self.mask, 
            self.keypoints, 
            self.boxes
        ]

    def _all_tensors(self: TransformInput) -> list[Tensor]:
        return [t for t in self._all_inputs() if self._istensor(t)]

    def __post_init__(self: TransformInput) -> None:
        assert self.image is not None, \
            'nectarml.vision.transforms require an image tensor.'
            
        if self._istensor(self.image):
            dims_msg = ('nectarml.vision.transforms expect input image '
                        'tensors to have 4 dimensions (B, C, H, W), but found '
                        'tensor with [{}] dimensions.')
            assert self.image.ndim == 4, dims_msg.format(self.image.ndim)
            
            if self._istensor(self.image2):
                assert self.image2.ndim == 4, dims_msg.format(self.image2.ndim)
            
        support_msg = ('nectarml.vision.transforms does not currently support '
                       '{}. Most operations will pass them through unaltered.')
        if self.mask is not None:      warn(support_msg.format('masks'))
        if self.boxes is not None:     warn(support_msg.format('boxes'))
        if self.keypoints is not None: warn(support_msg.format('keypoints'))
        
        for tensor in self._all_tensors():
            if tensor.requires_grad:
                warn('vision.transforms module found input tensor with '
                     'requires_grad=True. Autograd is not currently supported '
                     'for all transforms, and using them on tensors connected '
                     'to the compute graph may lead to unexpected results.')
        
    @classmethod
    def from_args(
        cls:    type[Self],
        args:   tuple[Tensor, ...],
        kwargs: dict[str, Tensor]
    ) -> TransformInput:
        '''
        Positional Convention:
        (image,)
        (image, mask)
        (image, image2, mask)
        '''
        if kwargs: return cls(**kwargs)
        
        match len(args):
            case 1: return cls(image=args[0])
            case 2: return cls(image=args[0], mask=args[1])
            case 3: return cls(image=args[0], image2=args[1], mask=args[2])
            case _: 
                raise ValueError(
                    f'Expected 1-3 positional args (image, image2, mask) '
                    f'or keyword args, got {len(args)}')

    def to_output(
        self:            TransformInput,
        original_args:   tuple,
        original_kwargs: dict
    ) -> Tensor | tuple[Tensor, ...]:
        if original_kwargs:
            output = tuple(
                {k: getattr(self, k) for k in original_kwargs}.values())
            if len(output) == 1: return output[0]
            return output

        match len(original_args):
            case 1: return self.image
            case 2: return self.image, self.mask
            case 3: return self.image, self.image2, self.mask
            
    def as_dict(self: TransformInput) -> dict[str, Any | None]:
        return {
            'image':     self.image,
            'image2':    self.image2,
            'mask':      self.mask,
            'boxes':     self.boxes,
            'keypoints': self.keypoints
        }

### UTILS ###

def hsv_adjust(
    input:      Tensor,
    hue_shift:  float = 0.0,
    saturation: float = 1.0,
    value:      float = 1.0
) -> Tensor:
    assert input.shape[1] == 3, \
        'hsv_adjust is only valid for 3-channel tensors (R, G, B).'
        
    if input.device == 'cuda':
        out_data = _nectarml.hsv_adjust(
            input._data_ptr, list(input.shape),
            hue_shift, saturation, value, input.dtype.cuda)
    else:
        img_array = (input.data).transpose((0, 2, 3, 1))
        
        hsv = np.vectorize(colorsys.rgb_to_hsv)(
            img_array[..., 0], img_array[..., 1], img_array[..., 2])
        
        h = (hsv[0] + hue_shift) % 1.0
        s = np.clip(hsv[1] * saturation, 0.0, 1.0)
        v = np.clip(hsv[2] * value, 0.0, 1.0)

        rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)
        out_data = np.clip(np.stack(rgb, axis=-1), 0, 255)
        out_data = out_data.transpose((0, 3, 1, 2)).astype(input.dtype.numpy)
        
    return Tensor._new(
        out_data, input.shape, input.dtype, input.device, input.requires_grad
    )

def apply_kernel_2d(image: Tensor, kernel: Tensor) -> Tensor:
    B, C, H, W = image.shape
    KH, KW     = kernel.shape
    kernel     = kernel.unsqueeze(0).unsqueeze(0)
    image_flat = image.reshape((B * C, 1, H, W))
    result     = F.conv2d(image_flat, kernel, padding=(KH//2, KW//2), groups=1)
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
    mode:  Literal[
        'horizontal', 'vertical', 'radial', 'elliptical'
    ] = 'horizontal',
    radial_method: Literal['corners', 'edges'] = 'corners',
    invert:        bool = False
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
            yy   = linspace(-1, 1, H).reshape((H, 1)).expand((H, W))
            xx   = linspace(-1, 1, W).reshape((1, W)).expand((H, W))
            dist = (xx**2 + yy**2).sqrt() / (2**0.5)
            mask = dist.clamp(0.0, 1.0).reshape((1, 1, H, W))
    
    mask = mask.clamp(0.0, 1.0)
    if invert: mask = 1 - mask
    return mask
        
