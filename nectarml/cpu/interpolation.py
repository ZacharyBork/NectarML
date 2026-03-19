import itertools
from typing import Literal

import numpy as np
from scipy.ndimage import zoom

ORDER_MAPPING = {
    'nearest':   0,
    'linear':    1,
    'bilinear':  1,
    'bicubic':   3,
    'trilinear': 1
}

MODE_NDIM = {
    'nearest':   None,
    'linear':    3,
    'bilinear':  4,
    'bicubic':   4,
    'trilinear': 5
}

### UPSAMPLE ###

def upsample(
    input: np.ndarray,
    out_sizes: tuple[int, ...],
    mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
) -> np.ndarray:
    mode_ndim = MODE_NDIM[mode]
    if mode_ndim is not None:
        assert input.ndim == mode_ndim, \
            f'Upsample mode [{mode}] expects input to have ndim={mode_ndim}.'
    zoom_factors = (1, 1) \
                 + tuple(o / i for o, i in zip(out_sizes, input.shape[2:]))
    return zoom(input, zoom_factors, order=ORDER_MAPPING[mode])

### BACKWARD ###

# NEAREST NEIGHBOR

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

# LINEAR/BILINEAR/TRILINEAR

def upsample_linear_backward_nd(
    grad_output: np.ndarray,
    in_sizes: tuple[int, ...]
) -> np.ndarray:
    n_spatial = len(in_sizes)
    out_sizes = grad_output.shape[2:]
    
    grad_input = np.zeros(
        grad_output.shape[:2] + in_sizes, dtype=grad_output.dtype)

    lows, highs, wt_highs = [], [], []
    for dim in range(n_spatial):
        in_float = np.arange(out_sizes[dim]) * (in_sizes[dim] / out_sizes[dim])
        low  = np.floor(in_float).astype(int)
        lows.append(low)
        highs.append(np.minimum(low + 1, in_sizes[dim] - 1))
        wt_highs.append(in_float - low)

    for corner in itertools.product(*[(0, 1)] * n_spatial):
        weight = np.ones(out_sizes, dtype=grad_output.dtype)
        idx = []
        for dim, is_high in enumerate(corner):
            wt = wt_highs[dim] if is_high else (1.0 - wt_highs[dim])
            coord = highs[dim] if is_high else lows[dim]
            
            shape = [1] * n_spatial
            shape[dim] = -1
            weight = weight * wt.reshape(shape)
            idx.append(coord.reshape(shape) * np.ones(out_sizes, dtype=int))
        
        idx = tuple(idx)
        
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                np.add.at(grad_input[b, c], idx, weight * grad_output[b, c])

    return grad_input

def upsample_linear_backward(
    grad_output: np.ndarray,
    in_size: int
) -> np.ndarray:
    return upsample_linear_backward_nd(grad_output, (in_size,))

def upsample_bilinear_backward(
    grad_output: np.ndarray,
    in_sizes: tuple[int, int]
) -> np.ndarray:
    return upsample_linear_backward_nd(grad_output, in_sizes)

def upsample_trilinear_backward(
    grad_output: np.ndarray,
    in_sizes: tuple[int, int, int]
) -> np.ndarray:
    return upsample_linear_backward_nd(grad_output, in_sizes)

# BICUBIC

def cubic_weight(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    t = np.abs(t)
    w = np.where(t <= 1,
        (a + 2) * t**3 - (a + 3) * t**2 + 1,
        np.where(t < 2,
            a * t**3 - 5*a * t**2 + 8*a * t - 4*a,
            0.0))
    return w

def upsample_bicubic_backward(
    grad_output: np.ndarray,
    in_sizes: tuple,
    a: float = -0.75
) -> np.ndarray:
    H_out, W_out = grad_output.shape[2], grad_output.shape[3]
    H_in, W_in = in_sizes

    grad_input = np.zeros(
        grad_output.shape[:2] + (H_in, W_in), dtype=grad_output.dtype)

    h_in_float = np.arange(H_out) * (H_in / H_out)
    w_in_float = np.arange(W_out) * (W_in / W_out)

    h_base = np.floor(h_in_float).astype(int)
    w_base = np.floor(w_in_float).astype(int)

    for b in range(grad_output.shape[0]):
        for c in range(grad_output.shape[1]):
            g = grad_output[b, c]

            for i in range(4):
                h_idx = np.clip(h_base + i - 1, 0, H_in - 1)
                wh = cubic_weight(h_in_float - (h_base + i - 1), a)[:, None]

                for j in range(4):
                    w_idx = np.clip(w_base + j - 1, 0, W_in - 1)
                    ww = cubic_weight(
                        w_in_float - (w_base + j - 1), a)[None, :]

                    h_grid = h_idx[:, None] * np.ones(W_out, dtype=int)
                    w_grid = np.ones(H_out, dtype=int)[:, None]*w_idx[None, :]

                    np.add.at(grad_input[b, c], (h_grid, w_grid), wh * ww * g)

    return grad_input

