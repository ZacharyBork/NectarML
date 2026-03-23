import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import float32
from nectarml.vision.transforms import Transform

class Sobel(Transform):
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

class Prewitt(Transform):
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

class Laplacian(Transform):
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
