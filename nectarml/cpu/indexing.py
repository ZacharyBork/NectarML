import numpy as np

def gather(input: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    return np.take_along_axis(input, index.astype(np.int32), axis=dim)

def scatter(
    input: np.ndarray, 
    dim:   int, 
    index: np.ndarray, 
    src:   np.ndarray
) -> np.ndarray:
    out = input.copy()
    np.put_along_axis(out, index.astype(np.int32), src, axis=dim)
    return out

def scatter_add(
    input: np.ndarray, 
    dim:   int, 
    index: np.ndarray, 
    src:   np.ndarray
) -> np.ndarray:
    out = input.copy()
    idx = [slice(None)] * input.ndim
    for i in range(index.shape[dim]):
        idx[dim] = index.take(i, axis=dim)
        np.add.at(out, tuple(idx), src.take(i, axis=dim))
    return out

def masked_fill(
    input: np.ndarray, 
    mask:  np.ndarray, 
    value: float
) -> np.ndarray:
    return np.where(mask, value, input)

def index_select(input: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    return np.take(input, index.astype(np.int32), axis=dim)

