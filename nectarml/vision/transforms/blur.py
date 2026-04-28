import warnings
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter

import nectarml.nn.functional as F
from nectarml.core     import Tensor, creation as T
from nectarml.typing   import float32
from nectarml.vision.transforms.transform import Transform
from nectarml.vision.transforms.common import TransformInput, apply_kernel_2d

class GaussianBlur(Transform):
    def __init__(
        self,
        kernel_size: int   | tuple[int,   int]   = (3, 7),
        sigma:       float | tuple[float, float] = 1.0,
        iterations:  int   | tuple[int,   int]   = 1,
        alpha:       float | tuple[float, float] = 1.0,
        p:           float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.sigma = (sigma, sigma) \
            if isinstance(sigma, float | int) else sigma
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        ks = self._ks
        
        x = T.linspace(0, ks-1, ks, dtype=float32, device=input.device) - (ks//2)
        xx = x.reshape((ks, 1)).expand((ks, ks))
        yy = x.clone().reshape((1, ks)).expand((ks, ks))
        
        kernel = (-(xx**2 + yy**2) / (2 * self._sigma**2)).exp()
        kernel = kernel / kernel.sum()
    
        blurred = input.clone()
        for _ in range(self._iters): 
            blurred = apply_kernel_2d(blurred, kernel)
        return ((1-self._alpha) * input + self._alpha*blurred).clamp(0.0, 1.0)

    def _build_parameters(self) -> None:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        self._ks = int(self.rng.choice(valid_sizes))
        self._sigma = self._random_in_range(self.sigma)
        self._iters = int(self._random_in_range(self.iterations))
        self._alpha = self._random_in_range(self.alpha)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class BoxBlur(Transform):
    def __init__(
        self,
        kernel_size: int   | tuple[int,   int]   = (3, 7),
        iterations:  int   | tuple[int,   int]   = 1,
        alpha:       float | tuple[float, float] = 1.0,
        p:           float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        kernel = (
            T.ones((self._ks, self._ks), device=input.device)
          / (self._ks * self._ks))
        
        blurred = input.clone()
        for _ in range(self._iters): 
            blurred = apply_kernel_2d(blurred, kernel)
        return ((1-self._alpha) * input + self._alpha*blurred).clamp(0.0, 1.0)
    
    def _build_parameters(self) -> None:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        self._ks = int(self.rng.choice(valid_sizes))
        self._iters = int(self._random_in_range(self.iterations))
        self._alpha = self._random_in_range(self.alpha)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class MotionBlur(Transform):
    def __init__(
        self,
        kernel_size: int   | tuple[int,   int]   = (3, 7),
        angle:       float | tuple[float, float] = (0.0, 360.0),
        iterations:  int   | tuple[int,   int]   = 1,
        alpha:       float | tuple[float, float] = 1.0,
        p:           float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.angle = (angle, angle) if isinstance(angle, int|float) else angle
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        
        ks = self._ks
        k  = T.zeros((ks, ks), device=input.device)
        center = ks // 2
        for i in range(ks):
            offset = i - center
            x = int(round(center + offset * np.cos(np.radians(self._angle))))
            y = int(round(center + offset * np.sin(np.radians(self._angle))))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if 0 <= x < ks and 0 <= y < ks: k[y, x] = 1.0
        total  = k.sum().item()
        kernel = k / total if total > 0 else k
        
        blurred = input.clone()
        for _ in range(self._iters): 
            blurred = apply_kernel_2d(blurred, kernel)
        return ((1-self._alpha) * input + self._alpha*blurred).clamp(0.0, 1.0)
    
    def _build_parameters(self) -> None:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        self._ks = int(self.rng.choice(valid_sizes))
        self._angle = int(self._random_in_range(self.angle))
        self._iters = int(self._random_in_range(self.iterations))
        self._alpha = self._random_in_range(self.alpha)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class MedianBlur(Transform):
    def __init__(
        self,
        kernel_size: int   | tuple[int,   int]   = (3, 7),
        alpha:       float | tuple[float, float] = 1.0,
        p:           float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha
        self.p = p
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        arr = input.cpu().numpy()
        result = np.stack([
            median_filter(arr[0, c], size=self._ks)
            for c in range(arr.shape[1])
        ])[np.newaxis]
        blurred = Tensor(result, dtype=input.dtype, device=input.device)
        return ((1-self._alpha) * input + self._alpha*blurred).clamp(0.0, 1.0)

    def _build_parameters(self) -> None:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        self._ks = int(self.rng.choice(valid_sizes))
        self._alpha = self._random_in_range(self.alpha)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomBlur(Transform):
    def __init__(
        self,
        kernel_size:         int   | tuple[int,   int]   = (3, 7),
        alpha:               float | tuple[float, float] = 1.0,
        iterations:          int   | tuple[int,   int]   = 1,
        motion_blur_angle:   float | tuple[float, float] = (0.0, 360.0),
        gaussian_blur_sigma: float | tuple[float, float] = 1.0,
        gaussian_blur:       bool  = True,
        box_blur:            bool  = True,
        motion_blur:         bool  = True,
        p:                   float = 0.5
    ) -> None:
        super().__init__(p=p)
        assert gaussian_blur + box_blur + motion_blur != 0, \
            'At least one blur type must be enabled.'

        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        self.motion_blur_angle = (motion_blur_angle, motion_blur_angle) \
            if isinstance(motion_blur_angle, int|float) else motion_blur_angle
        self.gaussian_blur_sigma = (gaussian_blur_sigma, gaussian_blur_sigma) \
            if isinstance(gaussian_blur_sigma, float | int) \
            else gaussian_blur_sigma
            
        self.gaussian_blur = gaussian_blur
        self.box_blur = box_blur
        self.motion_blur = motion_blur

    def forward(self, input: TransformInput) -> TransformInput:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        ks    = int(self.rng.choice(valid_sizes))
        iters = int(self._random_in_range(self.iterations))
        alpha = self._random_in_range(self.alpha)
        
        blur_fns = []
        if self.gaussian_blur: 
            sigma = self._random_in_range(self.gaussian_blur_sigma)
            blur_fns.append(GaussianBlur(ks, sigma, iters, alpha, p=1.0))
        if self.box_blur: blur_fns.append(BoxBlur(ks, iters, alpha, p=1.0))
        if self.motion_blur:
            angle = int(self._random_in_range(self.motion_blur_angle)) 
            blur_fns.append(MotionBlur(ks, angle, iters, alpha, p=1.0))
        blur_fn = self.rng.choice(blur_fns)

        return blur_fn.forward(input)

class Sharpen(Transform):
    def __init__(
        self,
        alpha:           float | tuple[float, float] = 1.0,
        iterations:      int   | tuple[int,   int]   = 1,
        kernel_size:     int   = 3,
        sigma:           float = 1.0,
        blur_iterations: int   = 1,
        method:          Literal['gaussian', 'box'] = 'gaussian',
        p:               float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
            
        match method:
            case 'box': 
                b = BoxBlur(kernel_size, blur_iterations, p=1.0)
            case 'gaussian': 
                b =  GaussianBlur(kernel_size, sigma, blur_iterations, p=1.0)
            case _: raise ValueError(f'Invalid blur method: {method}')
        
        b._build_parameters()
        self.blur = lambda x : b._transform(x)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        output = input.clone()
        for _ in range(self._iters):
            output = output + self._alpha * (output - self.blur(output))
            output = output.clamp(0.0, 1.0)
        return output

    def _build_parameters(self) -> None:
        self._alpha = self._random_in_range(self.alpha)
        self._iters = int(self._random_in_range(self.iterations))

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Emboss(Transform):
    def __init__(
        self,
        kernel_mode:   int  = 0,
        rotate_kernel: bool = False,
        gray_level:    int   | tuple[int,   int]   = 125,
        alpha:         float | tuple[float, float] = 1.0,
        p:             float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.gray_level = (gray_level, gray_level) \
            if isinstance(gray_level, int | float) else gray_level
        self.gray_level = tuple([i/255 for i in self.gray_level])
        self.alpha = (alpha, alpha) \
            if isinstance(alpha, int | float) else alpha
        
        match kernel_mode:
            case 0: 
                k = [[0,  1, 0],
                     [0,  0, 0],
                     [0, -1, 0]]
            case 1: 
                k = [[1, 0,  0],
                     [0, 0,  0],
                     [0, 0, -1]]
            case 2: 
                k = [[0, 0,  0],
                     [1, 0, -1],
                     [0, 0,  0]]
            case 3: 
                k = [[ 0, 0, 1],
                     [ 0, 0, 0],
                     [-1, 0, 0]]
            case _: 
                raise ValueError(
                    f'Kernel type expected values between 0-3 but found '
                    f'value: {kernel_mode}')
        
        k = np.array(k).astype(np.float32)
        if rotate_kernel: k = np.rot90(k)
        self.kernel = Tensor(k)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input        
        kernel   = self.kernel.to(input.device, input.dtype)
        gray     = input.mean(dim=1, keepdim=True)
        embossed = apply_kernel_2d(gray, kernel)
        embossed = (embossed + self._gray_level).clamp(0.0, 1.0)
        result   = F.cat([embossed, embossed, embossed], dim=1)
        return ((1-self._alpha) * input + self._alpha*result).clamp(0.0, 1.0)
        

    def _build_parameters(self) -> None:
        self._gray_level = self._random_in_range(self.gray_level)
        self._alpha      = self._random_in_range(self.alpha)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class UnsharpMask(Transform):
    def __init__(
        self,
        kernel_size: int   | tuple[int,   int]   = (3, 7),
        sigma:       float | tuple[float, float] = 1.0,
        iterations:  int   | tuple[int,   int]   = 1,
        strength:    float | tuple[float, float] = (0.5, 1.5),
        p:           float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        for size in self.kernel_size:
            assert size % 2 != 0, 'Kernel sizes must be an odd integer values.'
        self.sigma = (sigma, sigma) \
            if isinstance(sigma, float | int) else sigma
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        self.strength = (strength, strength) \
            if isinstance(strength, int | float) else strength
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        for _ in range(self._iters):
            blurred   = self._blur_fn._transform(input)
            sharpened = input + self._strength * (input - blurred)
            sharpened = sharpened.clamp(0.0, 1.0)
        return sharpened

    def _build_parameters(self) -> None:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        kernel_size   = int(self.rng.choice(valid_sizes))
        sigma         = float(self.rng.choice(self.sigma))
        self._blur_fn = GaussianBlur(kernel_size, sigma, p=1.0)
        self._blur_fn._build_parameters()
        
        self._iters    = int(self.rng.choice(self.iterations))
        self._strength = float(self.rng.choice(self.strength))

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
