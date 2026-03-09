import numpy as np

def reshape(input: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return input.reshape(shape)

def permute(input: np.ndarray, dims: tuple[int, ...] | None) -> np.ndarray:
    return np.transpose(input, axes=dims)
    
def expand(input: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(input, shape)

