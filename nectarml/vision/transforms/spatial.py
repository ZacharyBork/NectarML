from typing import Literal

import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import map_coordinates, gaussian_filter

import _nectarml
import nectarml.nn.functional as F
from nectarml.core        import Tensor
from nectarml.typing      import Size
from nectarml.vision.transforms.transform import Transform
from nectarml.vision.transforms.common    import TransformInput

### PADDING ###

class Pad(Transform):
    def __init__(
        self,
        padding:      int | tuple[int, ...],
        fill:         float = 0.0,
        padding_mode: Literal[
            'constant', 'reflect', 'replicate', 'circular'
        ] = 'constant',
        transform_mask: bool = True
    ) -> None:
        '''Applies padding to input image tensors.

        Args:
            padding      : The number of pixels to pad the input with. Can be a 
                           single integer, in which case the value will be used 
                           to, define the padding on all sides, or a tuple of 2 
                           integers, in which case it will be treated as 
                           (LR, TB), or a tuple of 4 integers for (L, R, T, B).
            fill         : The value (0-1) to fill the padded area with. Only 
                           used when `padding_mode` is `constant`.
            padding_mode : The padding mode to use. Options are:
                           1. `constant`  : Fill the padded area with a 
                                            constant grayscale value.
                           2. `reflect`   : Mirrors edge pixels in the padded
                                            area.
                           3. `Replicate` : Replicates the edge pixels into the
                                            padded area.
                           4. `circular`  : Wraps the opposite edges pixels to
                                            fill the padded area.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__()
        self.padding: tuple[int, ...] = None
        self.fill           = fill
        self.padding_mode   = padding_mode
        self.transform_mask = transform_mask
        
        self._init_padding(padding)
                
    def _init_padding(self, padding: int | tuple[int, ...]) -> None:
        if isinstance(padding, int): self.padding = (padding,) * 4
        elif isinstance(padding, tuple):            
            if len(padding) == 2:
                self.padding = (padding[0], padding[1], padding[0], padding[1])
            elif len(padding) == 4: self.padding = padding
            else: 
                raise ValueError(
                    'RandomCrop padding tuple must have either 2 [LT, RB] or '
                    '4 [L, T, R, B] values.')
        else: raise ValueError('Pad.padding must be int or tuple[int, ...].')
     
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return F.pad(input, self.padding, self.padding_mode, self.fill)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
### CROPPING ###
    
class _Crop(Transform):
    def __init__(
        self,
        size:           int | tuple[int, int],
        padding:        int | tuple[int, ...] | None = None,
        pad_if_needed:  bool  = False,
        fill:           float = 0.0,
        padding_mode:   Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant',
        p:              float = 1.0,
        transform_mask: bool  = True
    ) -> None:
        '''Abstract parent of cropping transform classes.

        Implements the actual tensor cropping logic, so that child classes can
        just define how the cropping should be applied.
        
        Args:
            size          : The size (H, W) to crop the inputs to. If only a 
                            single value is provided, it will be used for both 
                            height and width.
            padding       : The padding to apply (see vision.transform.Pad), or 
                            None for no padding.
            pad_if_needed : If True, and if the crop size is larger than the
                            input size, the input will be automatically padded 
                            to the crop size. If False, a RuntimeError will 
                            instead be raised if the transform recieves an 
                            input smaller than the crop size in any dimension.
            fill          : The fill value to use for the padded area if 
                            `padding_mode` is `constant`.
            padding_mode  : The padding mode to use if applicable. Options are:
                            1. `constant`  : Fill the padded area with a 
                                             constant grayscale value.
                            2. `reflect`   : Mirrors edge pixels in the padded
                                             area.
                            3. `Replicate` : Replicates the edge pixels into 
                                             the padded area.
                            4. `circular`  : Wraps the opposite edges pixels to
                                             fill the padded area.
            p               : The probability (0-1) of the effect being applied 
                              to any given input.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(p=p)
        if isinstance(size, int): self.size = (size, size)
        else: self.size = size
        
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        if padding is not None: self.pad = Pad(padding, fill, padding_mode)
        else: self.pad = None
        self.transform_mask = transform_mask
    
    def _validate_input_size(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        if H < self.size[0] or W < self.size[1]:
            if self.pad_if_needed:
                diff_h   = int(np.maximum(0, self.size[0] - H))
                diff_w   = int(np.maximum(0, self.size[1] - W))
                padding  = (diff_w, diff_h, diff_w, diff_h)
                self.pad = Pad(padding, self.fill, self.padding_mode)
                return self.pad(input)
            else:
                raise RuntimeError(
                    f'Input image size {input.shape[2:]} is greater than '
                    f'desired crop size: {self.size}')
        else: return input
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None:    return input
        if self.pad is None: return self._validate_input_size(input)
        else:                return self.pad(input)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class RandomCrop(_Crop):
    def __init__(
        self,
        size:           int | tuple[int, int],
        padding:        int | tuple[int, ...] | None = None,
        pad_if_needed:  bool  = False,
        fill:           float = 0.0,
        padding_mode: Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant',
        p:              float = 1.0,
        transform_mask: bool  = True
    ) -> None:
        '''Randomly crops input images to target size.

        Args:
            size          : The size (H, W) to crop the inputs to. If only a 
                            single value is provided, it will be used for both 
                            height and width.
            padding       : The padding to apply (see vision.transform.Pad), or 
                            None for no padding.
            pad_if_needed : If True, and if the crop size is larger than the
                            input size, the input will be automatically padded 
                            to the crop size. If False, a RuntimeError will 
                            instead be raised if the transform recieves an 
                            input smaller than the crop size in any dimension.
            fill          : The fill value to use for the padded area if 
                            `padding_mode` is `constant`.
            padding_mode  : The padding mode to use if applicable. Options are:
                            1. `constant`  : Fill the padded area with a 
                                             constant grayscale value.
                            2. `reflect`   : Mirrors edge pixels in the padded
                                             area.
                            3. `Replicate` : Replicates the edge pixels into 
                                             the padded area.
                            4. `circular`  : Wraps the opposite edges pixels to
                                             fill the padded area.
            p              : The probability (0-1) of the effect being applied 
                             to any given input.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(
            size, padding, pad_if_needed, fill, 
            padding_mode, p, transform_mask)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        out = super()._transform(input)
        return out[
            :, :, 
            self._offset_h:self._offset_h+self.size[0], 
            self._offset_w:self._offset_w+self.size[1]]

    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        max_offset = (input_shape[2]-self.size[0], input_shape[3]-self.size[1])
        self._offset_h = int(round(self._random_in_range((0, max_offset[0]))))
        self._offset_w = int(round(self._random_in_range((0, max_offset[1]))))
        
    def forward(self, input: TransformInput) -> TransformInput:        
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class CenterCrop(_Crop):
    def __init__(
        self,
        size:           int | tuple[int, int],
        padding:        int | tuple[int, ...] | None = None,
        pad_if_needed:  bool  = False,
        fill:           float = 0.0,
        padding_mode:   Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant',
        transform_mask: bool = True
    ) -> None:
        '''Center-crops input images to target size.

        Args:
            size          : The size (H, W) to crop the inputs to. If only a 
                            single value is provided, it will be used for both 
                            height and width.
            padding       : The padding to apply (see vision.transform.Pad), or 
                            None for no padding.
            pad_if_needed : If True, and if the crop size is larger than the
                            input size, the input will be automatically padded 
                            to the crop size. If False, a RuntimeError will 
                            instead be raised if the transform recieves an 
                            input smaller than the crop size in any dimension.
            fill          : The fill value to use for the padded area if 
                            `padding_mode` is `constant`.
            padding_mode  : The padding mode to use if applicable. Options are:
                            1. `constant`  : Fill the padded area with a 
                                             constant grayscale value.
                            2. `reflect`   : Mirrors edge pixels in the padded
                                             area.
                            3. `Replicate` : Replicates the edge pixels into 
                                             the padded area.
                            4. `circular`  : Wraps the opposite edges pixels to
                                             fill the padded area.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(
            size, padding, pad_if_needed, fill, 
            padding_mode, 1.0, transform_mask)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        out = super()._transform(input)

        offset = (input.shape[2]-self.size[0], input.shape[3]-self.size[1])
        offset_h = offset[0] // 2
        offset_w = offset[1] // 2
        return out[
            :, :, 
            offset_h:offset_h+self.size[0], 
            offset_w:offset_w+self.size[1]]
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class RandomResizedCrop(_Crop):
    def __init__(
        self,
        crop_size:     int | tuple[int, int],
        output_size:   int | tuple[int, int] | None = None,
        padding:       int | tuple[int, ...] | None = None,
        pad_if_needed: bool  = False,
        fill:          float = 0.0,
        padding_mode:  Literal[
            'constant', 'edge', 'reflect', 'symmetric'
        ] = 'constant',
        scaling_mode:  Literal['nearest', 'bilinear', 'bicubic'] = 'nearest',
        a:             float = -0.75,
        p:             float = 1.0,
        transform_mask: bool = True
    ) -> None:
        '''Randomly crops inputs, then scales result to target output size.

        Args:
            crop_size     : The size (H, W) to crop the inputs to before 
                            scaling. If only a single value is provided, it 
                            will be used for both height and width.
            output_size   : The size (H, W) to scale the image to after 
                            cropping is applied, or None to return the crop 
                            result unscaled. If only a single value is 
                            provided it will be used for both height and width.
            size          : The size (H, W) to crop the inputs to. If only a 
                            single value is provided, it will be used for both 
                            height and width.
            padding       : The padding to apply (see vision.transform.Pad), or 
                            None for no padding.
            pad_if_needed : If True, and if the crop size is larger than the
                            input size, the input will be automatically padded 
                            to the crop size. If False, a RuntimeError will 
                            instead be raised if the transform recieves an 
                            input smaller than the crop size in any dimension.
            fill          : The fill value to use for the padded area if 
                            `padding_mode` is `constant`.
            padding_mode  : The padding mode to use if applicable. Options are:
                            1. `constant`  : Fill the padded area with a 
                                             constant grayscale value.
                            2. `reflect`   : Mirrors edge pixels in the padded
                                             area.
                            3. `Replicate` : Replicates the edge pixels into 
                                             the padded area.
                            4. `circular`  : Wraps the opposite edges pixels to
                                             fill the padded area.
            scaling_mode  : The resampling mode to when scaling the crop 
                            result. Options are:
                            1. `nearest`  : Nearest neighbour resampling.
                            2. `bilinear` : Bilinear iterpolation.
                            3. `bicubic` : Bicubic interpolation.
            a              : Controls the smoothness of the kernel when 
                             `scaling_mode` is `bicubic`. Lower values will
                             produce sharper results, higher values will 
                             produce smoother results.
            p              : The probability (0-1) of the effect being applied 
                             to any given input.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(
            crop_size, padding, pad_if_needed, fill, 
            padding_mode, p, transform_mask)
        self.output_size = output_size
        self.scaling_mode = scaling_mode
        self.a = a
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        out = super()._transform(input)
        out = out[
            :, :, 
            self._offset_h:self._offset_h+self.size[0], 
            self._offset_w:self._offset_w+self.size[1]]
        out_size = input.shape[2:] if self.output_size is None \
              else self.output_size
        return F.upsample(
            out, size=out_size, mode=self.scaling_mode, a=self.a)
        
    def _build_parameters(self, input_shape: tuple[int, ...] | Size) -> None:
        max_offset = (input_shape[2]-self.size[0], input_shape[3]-self.size[1])
        self._offset_h = int(round(self._random_in_range((0, max_offset[0]))))
        self._offset_w = int(round(self._random_in_range((0, max_offset[1]))))
        
    def forward(self, input: TransformInput) -> TransformInput:        
        self._build_parameters(input.image.shape)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
 
### RESIZING ###
    
class Resize(Transform):
    def __init__(
        self,
        size:          int   | tuple[int,   ...] | None = None,
        scale_factor:  float | tuple[float, ...] | None = None,
        mode:          Literal['nearest', 'bilinear', 'bicubic'] = 'nearest',
        a:             float = -0.75,
        align_corners:  bool = False,
        transform_mask: bool = True
    ) -> None:
        '''Scales input to target size, or by scale factor.

        Output size of the resize operation can be defined in one of two
        ways:

        1. `size`: Directly defining the size of the spatial dimensions as a:
            - A Single integer (L) for 3-dimension tensors.
            - A tuple (H: int, W: int) for 4-dimension tensors.
            
        2. `scale_factor`: This acts as a multiplier to the spatial dimensions
            of the input tensor. So if you input a tensor with shape 
            (1, 3, 256, 256) and set `scale_factor` to (2.0, 2.0), the 
            output tensor would have shape (1, 3, 512, 512). `scale_factor` is
            defined as:
            
            - A Single float (L) for 3-dimension tensors.
            - A tuple (H: float, W: float) for 4-dimension tensors.

        NOTE: If both `size` and `scale_factor` are provided, Upsample will use 
        `scale_factor`.

        Args:
            size          : The desired output size.
            scale_factor  : The scale factor for the upsample.
            mode          : The resampling mode to use. Options are:
                            1. `nearest`  : Nearest neighbour resampling.
                            2. `bilinear` : Bilinear iterpolation.
                            3. `bicubic` : Bicubic interpolation.
            a              : Controls the smoothness of the kernel when 
                             `scaling_mode` is `bicubic`. Lower values will
                             produce sharper results, higher values will 
                             produce smoother results.
            align_corners  : If true, the corners of the input and the resized
                             result will be aligned, preserving their values.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__()
        self.size           = size
        self.scale_factor   = scale_factor
        self.mode           = mode
        self.a              = a
        self.align_corners  = align_corners
        self.transform_mask = transform_mask
            
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        return F.upsample(
            input, self.size, self.scale_factor, self.mode, 
            self.a, self.align_corners)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

### FLIPPING ###

class RandomHorizontalFlip(Transform):
    def __init__(
        self,
        p:             float = 0.5,
        transform_mask: bool = True
    ) -> None:
        '''Randomly flips inputs horizontally.
        
        Args:
            p              : The probability (0-1) of the effect being applied 
                             to any given input.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(p=p)
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        output = input[:, :, :, ::-1]
        return output

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomVerticalFlip(Transform):
    def __init__(
        self,
        p:              float = 0.5,
        transform_mask: bool  = True
    ) -> None:
        '''Randomly flips inputs vertically.
        
        Args:
            p              : The probability (0-1) of the effect being applied 
                             to any given input.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(p=p)
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        output = input[:, :, ::-1, :]
        return output
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Transpose(Transform):
    def __init__(
        self,
        p:              float = 0.5,
        transform_mask: bool  = True
    ) -> None:
        '''Randomly flips along both their horizontal, and vertical axes.
        
        Args:
            p              : The probability (0-1) of the effect being applied 
                             to any given input.
            transform_mask : If True, this transform will also be applied to 
                             the mask in the input TransformInput, if present.
        '''
        super().__init__(p=p)
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        output = input[:, :, ::-1, ::-1]
        return output
    
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

### ROTATING ###

class Rotate(Transform):
    def __init__(
        self,
        angle:          float = 90.0,
        fill_value:     float = 0.0,
        transform_mask: bool  = True,
        p:              float = 1.0
    ) -> None:
        super().__init__(p=p)
        self.angle = angle
        self.fill_value = fill_value
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        if input.device == 'cuda':
            out_data = _nectarml.rotate(
                input._data_ptr, list(input.shape),
                self.angle, self.fill_value, input.dtype.cuda)
        else:            
            in_data = input.data
            B, C, H, W = in_data.shape
            out_data = np.zeros_like(in_data)
            for b in range(B):
                for c in range(C):
                    out_data[b, c] = scipy_rotate(
                        in_data[b, c], self.angle, reshape=False,
                        order=1, mode='constant', cval=self.fill_value)
        
        return Tensor(out_data, input.shape, input.dtype, input.device)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class RandomRotation(Transform):
    def __init__(
        self,
        rotation_range: tuple[float, float] = (-180.0, 180.0),
        fill_value:     float = 0.0,
        transform_mask: bool  = True,
        p:              float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.rotation_range = rotation_range
        self.fill_value = fill_value
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        rotate = Rotate(self._angle, self.fill_value)
        return rotate(input)

    def _build_parameters(self) -> None:
        self._angle = self._random_in_range(self.rotation_range)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
    
class RandomRotate90(Transform):
    def __init__(
        self,
        mode:           Literal['90', '180', '270', '360'] = '360',
        fill_value:     float = 0.0,
        p:              float = 0.5,
        transform_mask: bool  = True
    ) -> None:
        super().__init__(p=p)
        self.fill_value = fill_value
        self.max_step = ['90', '180', '270', '360'].index(mode) + 1
        self.transform_mask = transform_mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        rotate = Rotate(self._step, self.fill_value)
        return rotate(input)
    
    def _build_parameters(self) -> None:
        self._step = 90 * int(round(self._random_in_range((0, self.max_step))))
        
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
  
### GRID SAMPLERS ###
  
class _GridSampleTransform(Transform):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)
        
    def _apply_flow(
        self,
        image: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray
    ) -> np.ndarray:
        C, H, W = image.shape
        result  = np.zeros_like(image)

        mode_map = {
            'reflect':  'reflect',
            'constant': 'constant',
            'nearest':  'nearest',
            'wrap':     'wrap'
        }
        mode = mode_map.get(self.border_mode, 'reflect')

        for c in range(C):
            result[c] = map_coordinates(
                image[c], [src_y.ravel(), src_x.ravel()],
                order=1, mode=mode, cval=self.fill
            ).reshape(H, W)

        return result
  
class RandomAffine(_GridSampleTransform):
    def __init__(
        self,
        degrees:     float | tuple[float, float] = (-45,   45),
        translate:   tuple[float, float] | None  = (-0.05, 0.05),
        scale:       tuple[float, float] | None  = (0.5,   2.0),
        shear:       float | tuple[float, float] | None = (-15.0, 15.0),
        border_mode: Literal[
            'reflect', 'constant', 'nearest', 'wrap'
        ] = 'reflect',
        fill:           float = 0.0,
        p:              float = 0.5,
        transform_mask: bool  = True
    ) -> None:
        super().__init__(p=p)
        self.degrees = (-degrees, degrees) \
            if isinstance(degrees, (int, float)) else degrees
        self.translate = translate
        self.scale = scale
        self.shear = (-shear, shear) \
            if isinstance(shear, (int, float)) else shear
        self.border_mode = border_mode
        self.fill = fill
        self.transform_mask = transform_mask

    def _compute_flow(
        self,
        H: int, W: int,
        angle: float,
        tx: float, ty: float,
        scale: float,
        shear: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = W / 2.0, H / 2.0
        angle_rad, shear_rad = np.radians(angle), np.radians(shear)

        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        tan_s = np.tan(shear_rad)

        M = np.array([
            [scale * cos_a, scale * (-sin_a + cos_a * tan_s), 0],
            [scale * sin_a, scale * ( cos_a + sin_a * tan_s), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        T_to_center = np.array(
            [[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float64)
        T_from_center = np.array(
            [[1,0, cx],[0,1, cy],[0,0,1]], dtype=np.float64)
        T_translate = np.array(
            [[1,0, tx],[0,1, ty],[0,0,1]], dtype=np.float64)

        M_full = T_translate @ T_from_center @ M @ T_to_center
        M_inv = np.linalg.inv(M_full)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        ones = np.ones_like(xx, dtype=np.float64)
        coords = np.stack([xx, yy, ones], axis=-1)
        flat = coords.reshape(-1, 3)
        warped = (M_inv @ flat.T).T
        
        return warped[:, 0].reshape(H, W), warped[:, 1].reshape(H, W)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        
        a = input.cpu().numpy()[0]
        src_x, src_y = self._compute_flow(
            H, W, self._angle, self._tx, 
            self._ty, self._scale, self._shear)
        result = self._apply_flow(a, src_x, src_y)
        return Tensor(
            result[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)

    def _build_parameters(self, H: int, W: int) -> None:
        self._angle = self._random_in_range(self.degrees)

        self._tx, self._ty = 0.0, 0.0
        if self.translate is not None:
            self._tx = self._random_in_range(
                (-self.translate[0] * W, self.translate[0] * W))
            self._ty = self._random_in_range(
                (-self.translate[1] * H, self.translate[1] * H))

        self._scale = 1.0 if self.scale is None else \
            self._random_in_range(self.scale)

        self._shear = 0.0 if self.shear is None else \
            self._random_in_range(self.shear)
            
    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,    # TODO: transform box coords
            keypoints = input.keypoints # TODO: transform keypoint coords
        )

class RandomPerspective(_GridSampleTransform):
    def __init__(
        self,
        distortion_scale: float = 0.5,
        border_mode: Literal[
            'reflect', 'constant', 'nearest', 'wrap'
        ] = 'reflect',
        fill:           float = 0.0,
        p:              float = 0.5,
        transform_mask: bool  = True
    ) -> None:
        super().__init__(p=p)
        self.distortion_scale = distortion_scale
        self.border_mode = border_mode
        self.fill = fill
        self.transform_mask = transform_mask

    def _compute_homography(
        self,
        src: np.ndarray,
        dst: np.ndarray
    ) -> np.ndarray:
        A = []
        for (sx, sy), (dx, dy) in zip(src, dst):
            A.append([-sx, -sy, -1,   0,   0,  0, dx*sx, dx*sy, dx])
            A.append([  0,   0,  0, -sx, -sy, -1, dy*sx, dy*sy, dy])
        A = np.array(A, dtype=np.float64)

        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def _compute_flow(
        self,
        H: int, W: int,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        H_mat = self._compute_homography(dst_pts, src_pts)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ones   = np.ones_like(xx, dtype=np.float64)
        coords = np.stack([xx, yy, ones], axis=-1)
        flat   = coords.reshape(-1, 3)

        warped = (H_mat @ flat.T).T
        w      = warped[:, 2:3]
        warped = warped[:, :2] / (w + 1e-8)

        return warped[:, 0].reshape(H, W), warped[:, 1].reshape(H, W)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        a = input.cpu().numpy()[0]
        src_x, src_y = self._compute_flow(H, W, self._src_pts, self._dst_pts)
        result = self._apply_flow(a, src_x, src_y)
        return Tensor(
            result[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)

    def _build_parameters(self, H: int, W: int) -> None:
        half_h = H * self.distortion_scale / 2
        half_w = W * self.distortion_scale / 2

        self._src_pts = np.array([
            [0,   0  ],
            [W-1, 0  ],
            [W-1, H-1],
            [0,   H-1]
        ], dtype=np.float64)

        uniform = self.rng.uniform
        self._dst_pts = np.array([
            [uniform(0, half_w),       uniform(0, half_h)      ],
            [uniform(W-1-half_w, W-1), uniform(0, half_h)      ],
            [uniform(W-1-half_w, W-1), uniform(H-1-half_h, H-1)],
            [uniform(0, half_w),       uniform(H-1-half_h, H-1)]
        ], dtype=np.float64)

    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,    # TODO: transform box coords
            keypoints = input.keypoints # TODO: transform keypoint coords
        )

class OpticalDistortion(_GridSampleTransform):
    def __init__(
        self,
        distort_limit: float | tuple[float, float] = 0.05,
        shift_limit:   float | tuple[float, float] = 0.05,
        border_mode:   Literal[
            'reflect', 'constant', 'nearest', 'wrap'
        ] = 'reflect',
        fill:          float = 0.0,
        p:             float = 0.5,
        transform_mask: bool = True
    ) -> None:
        super().__init__(p=p)
        self.distort_limit = (-distort_limit, distort_limit) \
            if isinstance(distort_limit, (int, float)) else distort_limit
        self.shift_limit = (-shift_limit, shift_limit) \
            if isinstance(shift_limit, (int, float)) else shift_limit
        self.border_mode = border_mode
        self.fill = fill
        self.transform_mask = transform_mask

    def _compute_flow(
        self,
        H: int, W: int,
        k:  float,
        dx: float,
        dy: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = W / 2.0, H / 2.0
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        xn, yn = (xx - cx) / cx, (yy - cy) / cy
        r2 = xn**2 + yn**2

        scale = 1.0 + k * r2
        src_x = cx + (xn * scale + dx) * cx
        src_y = cy + (yn * scale + dy) * cy
        
        return src_x.astype(np.float64), src_y.astype(np.float64)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        
        a = input.cpu().numpy()[0]
        src_x, src_y = self._compute_flow(H, W, self._k, self._dx, self._dy)
        result = self._apply_flow(a, src_x, src_y)
        return Tensor(
            result[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)

    def _build_parameters(self) -> None:
        self._k  = self._random_in_range(self.distort_limit)
        self._dx = self._random_in_range(self.shift_limit)
        self._dy = self._random_in_range(self.shift_limit)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,    # TODO: transform box coords
            keypoints = input.keypoints # TODO: transform keypoint coords
        )

class ElasticTransform(_GridSampleTransform):
    def __init__(
        self,
        alpha:         float | tuple[float, float] = (50.0, 150.0),
        sigma:         float | tuple[float, float] = (8.0, 12.0),
        border_mode:   Literal[
            'reflect', 'constant', 'nearest', 'wrap'
        ] = 'reflect',
        fill:          float = 0.0,
        p:             float = 0.5,
        transform_mask: bool = True
    ) -> None:
        super().__init__(p=p)
        self.alpha = (alpha, alpha) if isinstance(alpha, int|float) else alpha
        self.sigma = (sigma, sigma) if isinstance(sigma, int|float) else sigma
        self.border_mode = border_mode
        self.fill = fill
        self.transform_mask = transform_mask

    def _compute_flow(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        dx = self.rng.standard_normal((H, W)).astype(np.float32) * self._alpha
        dx = gaussian_filter(dx, sigma=self._sigma)
        
        dy = self.rng.standard_normal((H, W)).astype(np.float32) * self._alpha
        dy = gaussian_filter(dy, sigma=self._sigma)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        return (xx + dx).astype(np.float64), (yy + dy).astype(np.float64)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        a = input.cpu().numpy()[0]
        src_x, src_y = self._compute_flow(H, W)
        result = self._apply_flow(a, src_x, src_y)
        return Tensor(
            result[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)

    def _build_parameters(self) -> None:
        self._alpha = self._random_in_range(self.alpha)
        self._sigma = self._random_in_range(self.sigma)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,    # TODO: transform box coords
            keypoints = input.keypoints # TODO: transform keypoint coords
        )

class GridDistortion(_GridSampleTransform):
    def __init__(
        self,
        num_steps:     int = 5,
        distort_limit: float | tuple[float, float] = 0.3,
        border_mode:   Literal[
            'reflect', 'constant', 'nearest', 'wrap'
        ] = 'reflect',
        fill:          float = 0.0,
        p:             float = 0.5,
        transform_mask: bool = True
    ) -> None:
        super().__init__(p=p)
        self.num_steps = num_steps
        self.distort_limit = (-distort_limit, distort_limit) \
            if isinstance(distort_limit, int | float) else distort_limit
        self.border_mode = border_mode
        self.fill = fill
        self.transform_mask = transform_mask

    def _compute_flow(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        xx = np.arange(W, dtype=np.float64)
        yy = np.arange(H, dtype=np.float64)

        ctrl_x = np.linspace(0, W - 1, self.num_steps + 1)
        ctrl_y = np.linspace(0, H - 1, self.num_steps + 1)

        src_x = np.interp(xx, ctrl_x, self._stepsx)
        src_y = np.interp(yy, ctrl_y, self._stepsy)

        src_x, src_y = np.meshgrid(src_x, src_y, indexing='xy')
        return src_x, src_y

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        a = input.cpu().numpy()[0]
        src_x, src_y = self._compute_flow(H, W)
        result = self._apply_flow(a, src_x, src_y)
        return Tensor(
            result[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)

    def _build_parameters(self, H: int, W: int) -> None:
        stepsx = np.linspace(0, W - 1, self.num_steps + 1)
        stepsy = np.linspace(0, H - 1, self.num_steps + 1)

        for i in range(1, self.num_steps):
            rand_sx = self._random_in_range(self.distort_limit)
            rand_sy = self._random_in_range(self.distort_limit)
            stepsx[i] += rand_sx * W / self.num_steps
            stepsy[i] += rand_sy * H / self.num_steps

        self._stepsx = np.clip(stepsx, 0, W - 1)
        self._stepsy = np.clip(stepsy, 0, H - 1)

    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,    # TODO: transform box coords
            keypoints = input.keypoints # TODO: transform keypoint coords
        )

class Swirl(_GridSampleTransform):
    def __init__(
        self,
        strength:    float | tuple[float, float] = (1.0, 3.0),
        radius:      float | tuple[float, float] = (0.0, 360.0),
        center:      tuple[float, float] = (0.5, 0.5),
        border_mode: Literal[
            'reflect', 'constant', 'nearest', 'wrap'
        ] = 'reflect',
        fill:          float = 0.0,
        p:             float = 0.5,
        transform_mask: bool = True
    ) -> None:
        super().__init__(p=p)
        self.strength = (strength, strength) \
            if isinstance(strength, int | float) else strength
        self.radius = (-radius, radius) \
            if isinstance(radius, int | float) else radius
        self.center = center
        self.border_mode = border_mode
        self.fill = fill
        self.transform_mask = transform_mask

    def _compute_flow(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = self.center[1] * W, self.center[0] * H
        px, py = np.meshgrid(
            np.arange(W, dtype=np.float32), 
            np.arange(H, dtype=np.float32))

        dx, dy = px - cx, py - cy
        dist = np.sqrt(dx**2 + dy**2)
        angle = self._strength * np.exp(-dist / self._radius)
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        swirled_x = cos_a * dx - sin_a * dy
        swirled_y = sin_a * dx + cos_a * dy

        return swirled_x + cx, swirled_y + cy

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        a = input.cpu().numpy()[0]
        src_x, src_y = self._compute_flow(H, W)
        result = self._apply_flow(a, src_x, src_y)
        return Tensor(
            result[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)

    def _build_parameters(self) -> None:
        self._strength = self._random_in_range(self.strength)
        self._radius = self._random_in_range(self.radius)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) \
                        if self.transform_mask else input.mask,
            boxes     = input.boxes,    # TODO: transform box coords
            keypoints = input.keypoints # TODO: transform keypoint coords
        )

