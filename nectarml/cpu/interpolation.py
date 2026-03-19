import numpy as np
from scipy.ndimage import zoom

def upsample_nearest(
    input: np.ndarray,
    out_sizes: tuple
) -> np.ndarray:
    spatial_dims = input.shape[2:]
    zoom_factors = (1, 1) + tuple(
        o / i for o, i in zip(out_sizes, spatial_dims))
    return zoom(input, zoom_factors, order=0)

def upsample_nearest_backward(
    grad_output: np.ndarray,
    in_sizes: tuple
) -> np.ndarray:
    spatial_out = grad_output.shape[2:]
    spatial_in = in_sizes
    grad_input = np.zeros(
        grad_output.shape[:2] + spatial_in, dtype=grad_output.dtype)
    
    indices = tuple(
        np.minimum(
            (np.arange(o) * i // o).astype(int), i - 1)
        for o, i in zip(spatial_out, spatial_in))

    if len(spatial_in) == 1:
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                np.add.at(grad_input[b, c], indices[0], grad_output[b, c])
    elif len(spatial_in) == 2:
        idx_h = indices[0][:, None]
        idx_w = indices[1][None, :]
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                np.add.at(grad_input[b, c], (idx_h, idx_w), grad_output[b, c])
    elif len(spatial_in) == 3:
        idx_d = indices[0][:, None, None]
        idx_h = indices[1][None, :, None]
        idx_w = indices[2][None, None, :]
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                np.add.at(grad_input[b, c], 
                          (idx_d, idx_h, idx_w), grad_output[b, c])
    
    return grad_input
