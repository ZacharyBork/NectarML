from typing import Literal

import numpy as np

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import float32
from nectarml.vision.transforms import Transform

class Sobel(Transform[Tensor, Tensor]):
    def __init__(
        self,
        per_channel: bool = False,
        feldman: bool = False
    ) -> None:
        super().__init__()
        self.per_channel = per_channel
        
        if not feldman:
            self.sobel_x = Tensor([[[
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ]]], dtype=float32)
            self.sobel_y = Tensor([[[
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]]], dtype=float32)
        else:
            self.sobel_x = Tensor([[[
                [3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]
            ]]], dtype=float32)
            self.sobel_y = Tensor([[[
                [3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]
            ]]], dtype=float32)
        
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        norm = input / max_value
        kernel_x = self.sobel_x.to(input.device, input.dtype)
        kernel_y = self.sobel_y.to(input.device, input.dtype)

        outputs = []
        if not self.per_channel:
            gray = norm.mean(dim=1, keepdim=True)
            grad_fx = F.conv2d(gray, kernel_x, padding=1)
            grad_fy = F.conv2d(gray, kernel_y, padding=1)
            out = F.sqrt(grad_fx ** 2 + grad_fy ** 2 + 1e-6)
            outputs = [out]*3
        else:
            channels = norm.unbind(dim=1)
            for ch in channels:
                gray = ch.mean(dim=0, keepdim=True).unsqueeze(0)
                grad_fx = F.conv2d(gray, kernel_x, padding=1)
                grad_fy = F.conv2d(gray, kernel_y, padding=1)
                out = F.sqrt(grad_fx ** 2 + grad_fy ** 2 + 1e-6)
                outputs.append(out)
        
        return (F.cat(outputs, dim=1) * max_value).clamp(0.0, max_value)

class Prewitt(Transform[Tensor, Tensor]):
    def __init__(
        self,
        per_channel: bool = False
    ) -> None:
        super().__init__()
        self.per_channel = per_channel
        
        self.prewitt_x = Tensor([[[
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]]], dtype=float32)
        self.prewitt_y = Tensor([[[
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ]]], dtype=float32)

    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        norm = input / max_value
        kernel_x = self.prewitt_x.to(input.device, input.dtype)
        kernel_y = self.prewitt_y.to(input.device, input.dtype)

        outputs = []
        if not self.per_channel:
            gray = norm.mean(dim=1, keepdim=True)
            grad_fx = F.conv2d(gray, kernel_x, padding=1)
            grad_fy = F.conv2d(gray, kernel_y, padding=1)
            out = F.sqrt(grad_fx ** 2 + grad_fy ** 2 + 1e-6)
            outputs = [out]*3
        else:
            channels = norm.unbind(dim=1)
            for ch in channels:
                gray = ch.mean(dim=0, keepdim=True).unsqueeze(0)
                grad_fx = F.conv2d(gray, kernel_x, padding=1)
                grad_fy = F.conv2d(gray, kernel_y, padding=1)
                out = F.sqrt(grad_fx ** 2 + grad_fy ** 2 + 1e-6)
                outputs.append(out)
        
        return (F.cat(outputs, dim=1) * max_value).clamp(0.0, max_value)

class Laplacian(Transform[Tensor, Tensor]):
    def __init__(
        self,
        per_channel: bool = False
    ) -> None:
        super().__init__()
        self.per_channel = per_channel
        
        self.kernel = Tensor([[[
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ]]], dtype=float32)
        
    def forward(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        norm = input / max_value
        lap_kernel = self.kernel.to(input.device, input.dtype)

        outputs = []
        if not self.per_channel:
            gray = norm.mean(dim=1, keepdim=True)
            grad = F.conv2d(gray, lap_kernel, padding=1)
            out = F.sqrt(grad ** 2 + 1e-6)
            outputs = [out]*3
        else:
            channels = norm.unbind(dim=1)
            for ch in channels:
                gray = ch.mean(dim=0, keepdim=True).unsqueeze(0)
                grad = F.conv2d(gray, lap_kernel, padding=1)
                out = F.sqrt(grad ** 2 + 1e-6)
                outputs.append(out)
        
        return (F.cat(outputs, dim=1) * max_value).clamp(0.0, max_value)

class Dither(Transform[Tensor, Tensor]):
    def __init__(
        self,
        levels: int = 4,
        algorithm: Literal['floyd-steinberg'] = 'floyd-steinberg',
        per_channel: bool = True,
        from_channel: Literal['r', 'g', 'b'] = 'r'
    ) -> None:
        '''
        Fully sequential and pure CPU so very slow on large tensors. Likely 
        needs a dedicated CUDA kernel in the future.
        
        Big thanks to Christian Hill of scipython.com. This implementation was 
        adapted from his which can be found here:
            - https://scipython.com/blog/floyd-steinberg-dithering/
        '''
        super().__init__()
        self.levels = levels
        self.algorithm = algorithm
        self.per_channel = per_channel
        self.from_channel = from_channel

    def _get_new_val(self, old_val) -> np.ndarray:
        return np.round(old_val * (self.levels - 1)) / (self.levels - 1)

    def _floyd_steinberg(self, input: Tensor) -> Tensor:
        max_value = input.max().item()
        out = np.array(input.numpy().copy()) / max_value        
        
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

        out = np.clip(out, 0.0, 1.0) * max_value
        if not self.per_channel: 
            out = np.concatenate([out, out, out], axis=1)        
        return Tensor(out, out.shape, input.dtype, input.device)

    def forward(self, input: Tensor) -> Tensor:
        match self.algorithm:
            case 'floyd-steinberg': 
                return self._floyd_steinberg(input)
            case _: raise ValueError(
                f'Invalid Dither algortihm: {self.algorithm}')

