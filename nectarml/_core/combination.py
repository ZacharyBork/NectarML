from collections.abc import Callable, Sequence

import numpy as np

# def concatenate(
#     inputs: Sequence[np.ndarray], 
#     dim: int = 0
# ) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
#     out = np.concatenate(inputs, axis=dim)
#     def _backward(out_grad: np.ndarray) -> list[np.ndarray]:
#         sizes = [t.shape[dim] for t in inputs]
#         split_points = np.cumsum(sizes[:-1])
#         grads = np.split(out_grad, split_points, axis=dim)
#         return grads
#     return out, _backward

def concatenate(inputs: list[np.ndarray], dim: int = 0) -> np.ndarray:
    return np.concatenate(inputs, axis=dim)

# def stack(
#     inputs: Sequence[np.ndarray], 
#     dim: int = 0
# ) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
#     out = np.stack(inputs, axis=dim)
#     def _backward(out_grad: np.ndarray) -> list[np.ndarray]:
#         grads = np.split(out_grad, len(inputs), axis=dim)
#         return [np.squeeze(i, axis=dim) for i in grads]
#     return out, _backward

# def unstack(
#     input: np.ndarray, 
#     dim: int = 0
# ) -> tuple[list[np.ndarray], Callable[[np.ndarray], np.ndarray]]:
#     _split = np.split(input, input.shape[dim], axis=dim)
#     outputs = [np.squeeze(s, axis=dim) for s in _split]
#     def _backward(out_grads: list[np.ndarray]) -> np.ndarray:
#         return np.stack(out_grads, axis=dim)
#     return outputs, _backward

def unstack(tensor: np.ndarray, dim: int = 0) -> list[np.ndarray]:
    return np.split(tensor, tensor.shape[dim], axis=dim)

# def split(
#     input: np.ndarray, 
#     sizes: int | Sequence[int], 
#     dim: int = 0
# ) -> tuple[list[np.ndarray], Callable[[np.ndarray], np.ndarray]]:
#     outputs = np.split(input, sizes, axis=dim)
#     def _backward(out_grads: list[np.ndarray]) -> np.ndarray:
#         return np.concatenate(out_grads, axis=dim)
#     return outputs, _backward

def split(
    input: np.ndarray, 
    sizes: int | list[int], 
    dim: int = 0
) -> list[np.ndarray]:
    return np.split(input, sizes, axis=dim)


