from collections.abc import Sequence

import numpy as np

from nectarml import typing
from nectarml.tensor import Tensor
from nectarml.cuda import combination
from nectarml._core import combination as _combination

# from nectarml._core import combination

# def concatenate(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
#     out_data, _backward = combination.concatenate(
#         [t.data for t in inputs], dim=dim)
#     out = Tensor(out_data, out_data.shape, inputs[0].dtype, inputs[0].device,
#         inputs[0].requires_grad, tuple(inputs))
#     def _backward_hook():
#         grads = _backward(out.grad)
#         for tensor, grad in zip(inputs, grads):
#             if tensor.requires_grad:
#                 tensor.grad += grad
    
#     out._backward = _backward_hook
#     return out

# def cat(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
#     return concatenate(inputs, dim)


def concatenate(tensors: list[Tensor], dim: int = 0) -> Tensor:
    _devices = set([x.device for x in tensors])
    assert len(_devices) == 1, (
        f'nectarml.concatenate requires all Tensors be on the same device, '
        f'but found multiple devices: {list(_devices)}')
    
    _dtypes = set([x.dtype for x in tensors])
    assert len(_dtypes) == 1, (
        f'nectarml.concatenate requires all Tensors have the same dtype, '
        f'but found multiple dtypes: {list(_dtypes)}')
    
    device = _devices.pop()
    dtype = _dtypes.pop()
    shape = list(tensors[0].shape)
    requires_grad = tensors[0].requires_grad
    for i in tensors[1:]: 
        shape[dim] += list(i.shape)[dim]
        if i.requires_grad: requires_grad = True
    
    if device == 'cuda': data = combination.concatenate(tensors, dim)
    else: data = _combination.concatenate([t.data for t in tensors], dim=dim)
    
    out = Tensor(data, shape, dtype, device, requires_grad, tuple(tensors))

    # def _backward_hook():
    #     grads = _backward(out.grad)
    #     for tensor, grad in zip(inputs, grads):
    #         if tensor.requires_grad:
    #             tensor.grad += grad
    
    def _backward() -> None: pass
        # sizes = [t.shape[dim] for t in inputs]
        # split_points = np.cumsum(sizes[:-1])
        # grads = np.split(out_grad, split_points, axis=dim)
        # return grads
    
    out._backward = _backward
    return out
    
def cat(tensors: list[Tensor], dim: int = 0) -> Tensor:
    return concatenate(tensors, dim)

def stack(tensors: list[Tensor], dim: int = 0) -> Tensor:
    tensors = [x.unsqueeze(dim) for x in tensors]
    return concatenate(tensors, dim)
    

# def stack(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
#     out_data, _backward = combination.stack([t.data for t in inputs], dim=dim)
#     out = Tensor(out_data, out_data.shape, inputs[0].dtype, inputs[0].device,
#         inputs[0].requires_grad, tuple(inputs))
#     def _backward_hook():
#         grads = _backward(out.grad)
#         for tensor, grad in zip(inputs, grads):
#             if tensor.requires_grad:
#                 tensor.grad += grad
    
#     out._backward = _backward_hook
#     return out

# def unstack(input: Tensor, dim: int = 0) -> list[Tensor]:
#     out_data, _backward = combination.unstack(input.data, dim=dim)    
#     outputs = [
#         Tensor(i, i.shape, input.dtype, input.device,
#             input.requires_grad, _children=(input,))
#         for i in out_data]
    
#     _backward_called = False
#     def _backward_hook():
#         nonlocal _backward_called
#         if _backward_called: return
#         _backward_called = True
        
#         if input.requires_grad:
#             input.grad += _backward([o.grad for o in outputs])
    
#     for out in outputs:
#         out._backward = _backward_hook
#     return outputs

# def unbind(input: Tensor, dim: int = 0) -> list[Tensor]:
#     return unstack(input, dim)

# def split(
#     input: Tensor, 
#     sizes: int | Sequence[int], 
#     dim: int = 0
# ) -> list[Tensor]:
#     out_data, _backward = combination.split(input.data, sizes=sizes, dim=dim)    
#     outputs = [
#         Tensor(i, i.shape, input.dtype, input.device,
#             input.requires_grad, _children=(input,))
#         for i in out_data]

#     _backward_called = False
#     def _backward_hook():
#         nonlocal _backward_called
#         if _backward_called: return
#         _backward_called = True
#         if input.requires_grad:
#             input.grad += _backward([o.grad for o in outputs])
    
#     for out in outputs:
#         out._backward = _backward_hook
#     return outputs

# def chunk(input: Tensor, size: int, dim: int = 0) -> list[Tensor]:
#     assert size >= 1
#     chunk_size = int(np.ceil(input.shape[dim] / size))
#     return split(input, chunk_size, dim)

