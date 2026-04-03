import nectarml.functional as F
from nectarml.tensor import Tensor

### UTILS ###

def _apply_kernel_2d(image: Tensor, kernel: Tensor) -> Tensor:
    B, C, H, W = image.shape
    KH, KW = kernel.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    image_flat = image.reshape((B * C, 1, H, W))
    result = F.conv2d(image_flat, kernel, padding=(KH//2, KW//2), groups=1)
    return result.reshape((B, C, H, W))



