import numpy as np

def concatenate(inputs: list[np.ndarray], dim: int = 0) -> np.ndarray:
    return np.concatenate(inputs, axis=dim)

def unstack(tensor: np.ndarray, dim: int = 0) -> list[np.ndarray]:
    return np.split(tensor, tensor.shape[dim], axis=dim)

def split(
    input: np.ndarray, 
    sizes: int | list[int], 
    dim: int = 0
) -> list[np.ndarray]:
    return np.split(input, sizes, axis=dim)


