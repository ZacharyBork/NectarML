from nectarml.core import Tensor

def reshape(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return input.reshape(shape)

def view(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return input.view(shape)

def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    return input.flatten(start_dim, end_dim)

def squeeze(input: Tensor, dim: int | tuple[int, ...] | None) -> Tensor: 
    return input.squeeze(dim)
    
def unsqueeze(input: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    return input.unsqueeze(dim)

def transpose(input: Tensor, dim1: int, dim2: int) -> Tensor:
    return input.transpose(dim1, dim2)

def swapdims(input: Tensor, dim1: int, dim2: int) -> Tensor: 
    return input.transpose(dim1, dim2)

def permute(input: Tensor, dims: tuple[int, ...] | None) -> Tensor:
    return input.permute(dims)

def expand(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return input.expand(shape)

def broadcast_to(input: Tensor, shape: tuple[int, ...]) -> Tensor:
    return input.expand(shape)

def unfold(input: Tensor, dimension: int, size: int, step: int) -> Tensor:
    return input.unfold(dimension, size, step)

def flip(input: Tensor, dim: int) -> Tensor:
    return input.flip(dim)
