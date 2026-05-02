import warnings
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import nectarml.nn.functional as F
from nectarml.core        import Tensor, creation as T
from nectarml.typing      import float32
from nectarml.vision.transforms.transform import Transform
from nectarml.vision.transforms.common import TransformInput, apply_kernel_2d
from nectarml.vision.transforms.blur   import GaussianBlur

class Convolve(Transform):
    def __init__(
        self,
        kernel: list[list[int | float]] = [
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ],
        alpha: float | tuple[float, float] = 1.0,
        p:     float = 1.0
    ) -> None:
        '''Convolves input images with a specified kernel.

        NOTE: This transform can run natively on the GPU for CUDA tensors, and
        is much faster on the GPU.

        Args:
            kernel : The kernel to use for the convolution operation.
            alpha  : Blend strength (0-1) between the original input, and the
                     effect result.
            p      : The probability (0-1) of the effect being applied to any
                     given input.
        '''
        super().__init__(p=p)
        self.kernel = Tensor(kernel, dtype=float32)
        self.alpha = (alpha, alpha) if isinstance(alpha, int|float) else alpha
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        kernel  = self.kernel.to(input.device, input.dtype)
        result  = F.sqrt(apply_kernel_2d(input, kernel)**2 + 1e-6)
        blended = (1 - self._alpha) * input + self._alpha * result
        return blended.clamp(0.0, 1.0)
        
    def _build_parameters(self) -> None:
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

class Sobel(Transform):
    def __init__(
        self,
        per_channel: bool = False,
        scharr:      bool = False,
        p:          float = 1.0
    ) -> None:
        '''Applies a Sobel operator to input images. 

        Performs Sobel edge detection. Convolves input images twice with the 
        x/y kernels of a Sobel operator to isolate edges in the input images.

        NOTE: This transform can run natively on the GPU for CUDA tensors, and
        is much faster on the GPU.

        Args:
            per_channel : If True, the channels of the input will be split and
                          the Sobel filter will be applied to each channel
                          independently, then the results will be joined to 
                          form the output image.
            scharr      : If True, the kernels of a Scharr operator will be 
                          used in place of the traditional Sobel-Feldman 
                          kernels.
            p           : The probability (0-1) of the effect being applied to 
                          any given input.
        '''
        super().__init__(p=p)
        self.per_channel = per_channel
        
        if not scharr:
            kx = [[ 1,   0,  -1],
                  [ 2,   0,  -2],
                  [ 1,   0,  -1]]
            ky = [[ 1,   2,   1],
                  [ 0,   0,   0],
                  [-1,  -2,  -1]]
        else:
            kx = [[ 3,   0,  -3],
                  [10,   0, -10],
                  [ 3,   0,  -3]]
            ky = [[ 3,  10,   3],
                  [ 0,   0,   0],
                  [-3, -10,  -3]]
            
        self.sobel_x = Tensor(kx, dtype=float32)
        self.sobel_y = Tensor(ky, dtype=float32)
      
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        kernel_x = self.sobel_x.to(input.device, input.dtype)
        kernel_y = self.sobel_y.to(input.device, input.dtype)

        outputs = []
        if not self.per_channel:
            gray = input.mean(dim=1, keepdim=True)
            out  = F.sqrt(
                apply_kernel_2d(gray, kernel_x)**2 
              + apply_kernel_2d(gray, kernel_y)**2
              + 1e-6)
            outputs = [out]*3
        else:
            channels = input.unbind(dim=1)
            for ch in channels:
                gray = ch.mean(dim=0, keepdim=True).unsqueeze(0)
                out  = F.sqrt(
                apply_kernel_2d(gray, kernel_x)**2 
              + apply_kernel_2d(gray, kernel_y)**2
              + 1e-6)
                outputs.append(out)
        
        return F.cat(outputs, dim=1).clamp(0.0, 1.0)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Prewitt(Transform):
    def __init__(
        self,
        per_channel: bool  = False,
        p:           float = 1.0
    ) -> None:
        '''Applies a Prewitt operator to input images. 

        Performs Prewitt edge detection. Convolves input images twice with the 
        x/y kernels of a Prewitt operator to isolate edges in the input images.

        NOTE: This transform can run natively on the GPU for CUDA tensors, and
        is much faster on the GPU.

        Args:
            per_channel : If True, the channels of the input will be split and
                          the Prewitt filter will be applied to each channel
                          independently, then the results will be joined to 
                          form the output image.
            p           : The probability (0-1) of the effect being applied to 
                          any given input.
        '''
        super().__init__(p=p)
        self.per_channel = per_channel
        
        self.prewitt_x = Tensor([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ], dtype=float32)
        self.prewitt_y = Tensor([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ], dtype=float32)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        kernel_x = self.prewitt_x.to(input.device, input.dtype)
        kernel_y = self.prewitt_y.to(input.device, input.dtype)

        outputs = []
        if not self.per_channel:
            gray = input.mean(dim=1, keepdim=True)
            out  = F.sqrt(
                apply_kernel_2d(gray, kernel_x)**2 
              + apply_kernel_2d(gray, kernel_y)**2
              + 1e-6)
            outputs = [out]*3
        else:
            channels = input.unbind(dim=1)
            for ch in channels:
                gray = ch.mean(dim=0, keepdim=True).unsqueeze(0)
                out  = F.sqrt(
                apply_kernel_2d(gray, kernel_x)**2 
              + apply_kernel_2d(gray, kernel_y)**2
              + 1e-6)
                outputs.append(out)
        
        return F.cat(outputs, dim=1).clamp(0.0, 1.0)

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Laplacian(Transform):
    def __init__(
        self,
        per_channel: bool = False,
        p:          float = 1.0
    ) -> None:
        '''Applies a discrete Laplace operator to input images. 

        Performs Laplacian edge detection. Convolves input images with a 2D
        Laplacian kernel to isolate high-frequency detail in the input images.

        NOTE: This transform can run natively on the GPU for CUDA tensors, and
        is much faster on the GPU.

        Args:
            per_channel : If True, the channels of the input will be split and
                          the Laplacian filter will be applied to each channel
                          independently, then the results will be joined to 
                          form the output image.
            p           : The probability (0-1) of the effect being applied to 
                          any given input.
        '''
        super().__init__(p=p)
        self.per_channel = per_channel
        
        self.kernel = Tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=float32)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        kernel = self.kernel.to(input.device, input.dtype)

        outputs = []
        if not self.per_channel:
            gray    = input.mean(dim=1, keepdim=True)
            out     = F.sqrt(apply_kernel_2d(gray, kernel) ** 2 + 1e-6)
            outputs = [out]*3
        else:
            channels = input.unbind(dim=1)
            for ch in channels:
                gray = ch.mean(dim=0, keepdim=True).unsqueeze(0)
                out  = F.sqrt(apply_kernel_2d(gray, kernel) ** 2 + 1e-6)
                outputs.append(out)
        
        return F.cat(outputs, dim=1).clamp(0.0, 1.0)
        
    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Dither(Transform):
    def __init__(
        self,
        levels:       int = 4,
        algorithm:    Literal['floyd-steinberg'] = 'floyd-steinberg',
        per_channel:  bool = True,
        from_channel: Literal['r', 'g', 'b'] = 'r',
        p:            float = 1.0
    ) -> None:
        '''Applies dithering to input images.

        Big thanks to Christian Hill of scipython.com. This implementation was 
        adapted from his which can be found here:
            - https://scipython.com/blog/floyd-steinberg-dithering/

        NOTE: This transform runs fully sequentially and purely on the CPU so 
        it is very slow on large tensors. It will likely recieve a dedicated 
        CUDA kernel in the future.

        Args:
            levels       : Number of quantization levels to apply.
            algorithm    : Not currently used, here for the addition of more
                           dithering algorithms in the future.
            per_channel  : If True, the dithering effect will be applied to 
                           each channel independently. If False, it will 
                           instead be applied based on a single channel of
                           the input image only. Note that the resulting output 
                           tensor will have the same number of channels as the 
                           input tensor, the values will just be duplicated per 
                           channel to create the grayscale effect.
            from_channel : The channel to apply the dithering based on if 
                           `per_channel` is False.
            p            : The probability (0-1) of the effect being applied to 
                           any given input.
        '''
        super().__init__(p=p)
        assert levels > 1, 'Dither "levels" must be > 1.'
        self.levels       = levels
        self.algorithm    = algorithm
        self.per_channel  = per_channel
        self.from_channel = from_channel

    def _get_new_val(self, old_val) -> np.ndarray:
        return np.round(old_val * (self.levels - 1)) / (self.levels - 1)

    def _floyd_steinberg(self, input: Tensor) -> Tensor:
        out = input.numpy().copy()       
        
        if not self.per_channel: 
            out = out[:, ['r', 'g', 'b'].index(self.from_channel), :, :]
            out = out[np.newaxis]
        B, C, H, W = out.shape
        
        for b in range(B):
            for c in range(C):
                for y in range(H):
                    for x in range(W):
                        old_val = out[b, c, y, x].copy()
                        new_val = self._get_new_val(old_val)
                        out[b, c, y, x] = new_val
                        err = old_val - new_val

                        if x < W - 1:
                            out[b, c, y, x+1] += err * 7/16
                        if y < H - 1:
                            if x > 0:
                                out[b, c, y+1, x-1] += err * 3/16
                            out[b, c, y+1, x] += err * 5/16
                            if x < W - 1:
                                out[b, c, y+1, x+1] += err / 16

        out = np.clip(out, 0.0, 1.0)
        if not self.per_channel: 
            out = np.concatenate([out, out, out], axis=1)        
        return Tensor(out, out.shape, input.dtype, input.device)

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        match self.algorithm:
            case 'floyd-steinberg': 
                return self._floyd_steinberg(input)
            case _: raise ValueError(
                f'Invalid Dither algortihm: {self.algorithm}')

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Halftone(Transform):
    def __init__(
        self,
        cell_size:  int = 10,
        foreground: tuple[int, int, int]  = (0, 0, 0),
        background: tuple[int, int, int]  = (255, 255, 255),
        alpha:      float | tuple[float, float] = 0.5,
        p:          float = 1.0
    ) -> None:
        '''Applies a halftone filter to input images.

        NOTE: This transform runs purely on the CPU regardless on input tensor
        device so it can be slow on large tensors.

        Args:
            cell_size   : The size of each individual cell in the pattern.
            foreground  : The color (RGB, 0-255) of the foreground.
            background  : The color (RGB, 0-255) of the background.
            alpha       : Blend strength (0-1) between the original input, and 
                          the effect result. Can be a single floating point 
                          number in range [0:1], or a tuple of floats in the 
                          same range, in which case, a random value between the 
                          first and second value will be chosen each time the 
                          transform is called.
            p           : The probability (0-1) of the effect being applied to 
                          any given input.
        '''
        super().__init__(p=p)
        self.cell_size  = cell_size
        self.foreground = tuple([i/255 for i in foreground])
        self.background = tuple([i/255 for i in background])
        self.alpha = (alpha, alpha) if isinstance(alpha, float|int) else alpha

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input        
        arr = input.cpu().numpy()[0]
        C, H, W = arr.shape
        
        gray = arr.mean(axis=0)
        
        cs = self.cell_size
        rows, cols = H // cs, W // cs
        
        out = np.ones((C, H, W), dtype=np.float32)
        for c in range(C): out[c] = self.background[c]
        
        fg = np.array(self.foreground, dtype=np.float32)
        
        cy, cx     = np.ogrid[:cs, :cs]
        center     = cs / 2.0
        dist       = np.sqrt((cx - center)**2 + (cy - center)**2)
        max_radius = center * 0.95
        
        for r in range(rows):
            for c in range(cols):
                y0 = r  * cs
                x0 = c  * cs
                y1 = y0 + cs
                x1 = x0 + cs
                
                brightness = gray[y0:y1, x0:x1].mean()
                radius = (1.0 - brightness) * max_radius
                mask   = dist <= radius
                
                for c in range(C):
                    cell       = out[c, y0:y1, x0:x1]
                    cell[mask] = fg[c]
                    out[c, y0:y1, x0:x1] = cell
        
        result = Tensor(out[np.newaxis].astype(input.dtype.numpy), 
            dtype=input.dtype, device=input.device)
        return ((1-self._alpha) * input + self._alpha*result).clamp(0.0, 1.0)

    def _build_parameters(self) -> None:
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

class Kuwahara(Transform):
    def __init__(
        self,
        radius: int   = 7,
        p:      float = 1.0
    ) -> None:
        '''Applies a Kuwahara filter to input images.

        Adaptive noise reduction filter which is not generally all that useful
        for reducing noise, save for very particular circumstances. It does,
        however, produce a painterly stylized effect which is quite nice.

        See here: https://en.wikipedia.org/wiki/Kuwahara_filter

        Args:
            radius : The filter radius. Larger values will smooth over larger
                     areas.
            p      : The probability (0-1) of the effect being applied to 
                     any given input.
        '''
        super().__init__(p=p)
        self.radius = radius
      
    def make_kernel(self, row_slice: slice, col_slice: slice) -> Tensor:
        size = 2 * self.radius + 1
        k    = T.zeros((size, size))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k[row_slice, col_slice] = 1.0
        return k / (k.sum().item() + self._epsilon)
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        B, C, H, W = input.shape
        r = self.radius
        h = r + 1

        kernels = F.stack([
            self.make_kernel(slice(0, h), slice(0, h)),
            self.make_kernel(slice(0, h), slice(r, None)),
            self.make_kernel(slice(r, None), slice(0, h)), 
            self.make_kernel(slice(r, None), slice(r, None)),
        ]).to(input.device, input.dtype)

        kernels = kernels.unsqueeze(1)

        x_flat = input.reshape((B * C, 1, H, W))
        x2_flat = x_flat ** 2

        means = F.conv2d(x_flat,  kernels, padding=r, groups=1)
        means2 = F.conv2d(x2_flat, kernels, padding=r, groups=1)
        variances = means2 - means ** 2

        best = variances.argmin(dim=1, keepdim=True)        
        result = means.gather(dim=1, index=best).squeeze(1)

        return result.reshape((B, C, H, W))

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Pixelate(Transform):
    def __init__(
        self,
        block_size: int | tuple[int, int] = (6, 12),
        p:          float = 1.0
    ) -> None:
        '''Applies a pixelation filter to input images.
        
        Args:
            block_size : The size of each block of pixels. Can be a single 
                         integer, or a tuple of two integers, in which case, a
                         random value between the two (inclusive) will be 
                         chosen each time the transform is called.
            p          : The probability (0-1) of the effect being applied to 
                         any given input.
        '''
        super().__init__(p=p)
        self.block_size = (block_size, block_size) \
            if isinstance(block_size, int) else block_size
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        down = F.avg_pool2d(input, self._block_size, self._block_size)
        return F.upsample(down, size=(H, W), mode='nearest')
        
    def _build_parameters(self) -> None:
        self._block_size = int(self._random_in_range(self.block_size))

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class AsciiRender(Transform):
    def __init__(
        self,
        block_size:     int = 12,
        font:           str | None = None,
        charset:        Literal['minimal', 'dense', 'block'] = 'dense',
        custom_charset: str | None = None,
        sample_color:   bool = True,
        p:              float = 1.0
    ) -> None:
        '''Renders input images using an ascii character set.
        
        Args:
            block_size     : The size of each character. Smaller values produce
                             denser results.
            font           : System path to the font to use when rendering, or
                             `None` to use the default font. Works best with
                             monospace fonts.
            charset        : What charset to use when rendering the image.
                             Options are [`minimal`, `dense`, `block`].
            custom_charset : A string defining the character set to use when
                             rendering the image. Characters aranged from 
                             darkest to lightest. For example:
                             `@%#*+=-:. ` or `█▓▒░ `
            sample_color   : If true, the characters in the final image will 
                             sample the color of the input image. If False,
                             they will be white with a black background.
            p              : The probability (0-1) of the effect being applied 
                             to any given input.
        '''
        super().__init__(p=p)
        assert block_size > 0, 'AsciiRender "block_size" must be > 0.'
        self.block_size   = block_size
        self.sample_color = sample_color
        self.font_path    = font

        if custom_charset is None:
            match charset:
                case 'block': self.charset = '█▓▒░ '
                case 'minimal': self.charset = '@%#*+=-:. '
                case 'dense': 
                    self.charset = (
                        '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunx'
                        'rjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. ')
        else: self.charset = custom_charset

    def _get_font(self) -> ImageFont.FreeTypeFont:
        lo, hi = 1, self.block_size * 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            font = ImageFont.truetype(self.font_path, mid) \
                if self.font_path else ImageFont.load_default(mid)
            bbox = font.getbbox('A')
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if max(w, h) <= self.block_size: lo = mid
            else: hi = mid - 1
            
        return ImageFont.truetype(self.font_path, lo) \
            if self.font_path else ImageFont.load_default(lo)

    def _to_ascii(self, input: Tensor) -> list[str]:
        _, _, H, W = input.shape
        width  = max(1, W // self.block_size)
        height = max(1, H // self.block_size)

        gray = input.mean(dim=1, keepdim=True)
        gray = F.upsample(gray, size=(height, width), mode='nearest')

        pixels = gray.cpu().numpy()[0, 0]
        lo, hi = np.percentile(pixels, 2), np.percentile(pixels, 98)
        pixels = np.clip((pixels - lo) / (hi - lo + self._epsilon), 0.0, 1.0)
        
        indices = np.clip(
            (pixels * (len(self.charset) - 1)).astype(np.int32),
            0, len(self.charset) - 1)
        char_array = np.array(list(self.charset))[indices]

        return [''.join(row) for row in char_array]

    def _to_image(self, lines: list[str], original: Tensor) -> np.ndarray:
        font = self._get_font()
        bbox = font.getbbox('A')
        char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        rows, cols = len(lines), max(len(line) for line in lines)

        if self.sample_color:
            small  = F.upsample(original, size=(rows, cols), mode='nearest')
            small  = small.cpu().numpy()[0]
            lo, hi = small.min(), small.max()
            small  = ((small - lo) / (hi-lo + self._epsilon) * 255)
            small  = small.astype(np.int32)

        img = Image.new('RGB', (char_w * cols, char_h * rows), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                if self.sample_color: r, g, b = small[:, row, col].tolist()
                else: r, g, b = 255, 255, 255
                x, y = col * char_w, row * char_h
                if char == ' ':
                    draw.rectangle([x, y, x+char_w, y+char_h], fill=(r, g, b))
                else: draw.text((x, y), char, fill=(r, g, b), font=font)

        return np.array(img)
                
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, _, H, W = input.shape
        
        lines = self._to_ascii(input)
        arr   = self._to_image(lines, input)
        arr   = arr.transpose(2, 0, 1).astype(np.float32) / arr.max().item()
        out   = Tensor(arr[np.newaxis].astype(input.dtype.numpy), 
                       dtype=input.dtype, device=input.device)
        return F.upsample(out, size=(H, W), mode='nearest')

    def forward(self, input: TransformInput) -> TransformInput:
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class DifferenceOfGaussians(Transform):
    def __init__(
        self,
        kernel_size:int   | tuple[int,   int]   = 7,
        sigma1:     float | tuple[float, float] = 1.5,
        sigma2:     float | tuple[float, float] = 1.0,
        iterations: int   | tuple[int,   int]   = 1,
        alpha:      float | tuple[float, float] = 1.0,
        phi:        float | tuple[float, float] = 0.0,
        tau:        float | tuple[float, float] = 0.99,
        threshold:  float | None = None,
        invert:     bool  = False,
        gray:       bool  = True,
        p:          float = 1.0
    ) -> None:
        '''Applies an extended difference of Gaussian filter to input images.
        
        Blurs image twice using Gaussian filters with different sigmas, then
        takes the difference. Can be used for high quality, reasonably 
        performant edge detection, as well as a number of stylized effects
        depending on the input parameters.
        
        Reference:
            - https://en.wikipedia.org/wiki/Difference_of_Gaussians
            - Acerola, This is the Difference of Gaussians:
              https://www.youtube.com/watch?v=5EuYKEvugLU
            
        Args:
            kernel_size : The Gaussian kernel size.
            sigma1      : The sigma of the first Gaussian.
            sigma2      : The sigma of the second Gaussian.
            iterations  : The number of blurring iterations.
            alpha       : Blend strength (0-1) between the original input, and 
                          the effect result. Can be a single floating point 
                          number in range [0:1], or a tuple of floats in the 
                          same range, in which case, a random value between the 
                          first and second value will be chosen each time the 
                          transform is called.
            phi         : If `threshold` is not None, and this value is 0.0,
                          the difference result will be run through a binary
                          threshold function based on the value of `threshold`.
                          If `threshold` is not None and this value is greater 
                          than 0.0, any values below the threshold will instead
                          be run through a hyperbolic tangent function with 
                          `phi` controlling the falloff.
            tau         : Scale factor for the second Gaussian.
            threshold   : The value threshold at which to transition between
                          colors, or None to not apply a threshold function.
            invert      : Swaps the two Gaussians.
            gray        : If True, the effect will be applied to a grayscale
                          version of the image. If False, it will be applied to
                          all channels. Note that the resulting output tensor
                          will have the same number of channels as the input
                          tensor, the values will just be duplicated per 
                          channel to create the grayscale effect.
            p           : The probability (0-1) of the effect being applied 
                          to any given input.
        '''
        super().__init__(p=p)
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        self.sigma1 = (sigma1, sigma1) \
            if isinstance(sigma1, float|int) else sigma1
        self.sigma2 = (sigma2, sigma2) \
            if isinstance(sigma2, float|int) else sigma2
        self.iterations = (iterations, iterations) \
            if isinstance(iterations, int) else iterations
        
        self.alpha = (alpha, alpha) if isinstance(alpha, int|float) else alpha
        self.phi   = (phi,   phi)   if isinstance(phi,   int|float) else phi
        self.tau   = (tau,   tau)   if isinstance(tau,   int|float) else tau
        
        self.threshold = threshold
        self.invert    = invert
        self.gray      = gray
        
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        if self.gray: input = input.mean(dim=1, keepdim=True)

        g1 = self._g1._transform(input)
        g2 = self._g2._transform(input) 

        if self.invert: diff = g2 - self._tau * g1
        else:           diff = g1 - self._tau * g2

        diff_min = diff.min().item()
        diff_max = diff.max().item()
        diff     = (diff - diff_min) / (diff_max - diff_min + self._epsilon)

        if self.threshold is not None:
            if self._phi == 0.0:
                diff = F.where(diff >= self.threshold, T.ones_like(diff), 0.0)
            else:
                diff = F.where(
                    diff >= self.threshold,
                    T.ones_like(diff),
                    1 + F.tanh(self._phi * (diff - self.threshold)))

        result = (1 - self._alpha) * input + self._alpha * diff
        if self.gray: result = F.cat([result]*3, dim=1)
        return result.clamp(0.0, 1.0)
            
    def _build_parameters(self) -> None:
        valid_sizes = [
            i for i in range(self.kernel_size[0], self.kernel_size[1]+1)
            if i % 2 != 0]
        _ks         = int(self.rng.choice(valid_sizes))
        _iterations = int(self._random_in_range(self.iterations))
        _sigma1     = self._random_in_range(self.sigma1)
        _sigma2     = self._random_in_range(self.sigma2)
        
        
        self._g1 = GaussianBlur(_ks, _sigma1, _iterations, p=1)
        self._g2 = GaussianBlur(_ks, _sigma2, _iterations, p=1)
        
        self._g1._build_parameters()
        self._g2._build_parameters()
        
        self._alpha = self._random_in_range(self.alpha)
        self._phi   = self._random_in_range(self.phi)
        self._tau   = self._random_in_range(self.tau)

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
