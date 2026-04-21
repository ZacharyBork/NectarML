from nectarml.tensor       import Tensor
from nectarml.cuda         import combination as cuda
from nectarml.cpu          import combination as cpu
from nectarml.amp.autocast import autocast_state
from nectarml.amp.utils    import get_promotion_dtype

def concatenate(tensors: list[Tensor], dim: int = 0) -> Tensor:
    _devices = set([x.device for x in tensors])
    assert len(_devices) == 1, (
        f'nectarml.concatenate requires all Tensors be on the same device, '
        f'but found multiple devices: {list(_devices)}')
    
    _dtypes = set([x.dtype for x in tensors])
    state   = autocast_state()
    if state.enabled and state.context == 'cuda':
        if len(_dtypes) != 1:
            promote_type = get_promotion_dtype(_dtypes)
            tensors = [i.to(dtype=promote_type) for i in tensors]
    else:        
        assert len(_dtypes) == 1, (
            f'nectarml.concatenate requires all Tensors have the same dtype, '
            f'but found multiple dtypes: {list(_dtypes)}')
        
    device = _devices.pop()
    dtype  = _dtypes.pop()
    shape  = list(tensors[0].shape)
    dim    = Tensor._normalize_dim(dim, tensors[0].ndim)
    _grad  = tensors[0].requires_grad
    for i in tensors[1:]: 
        shape[dim] += list(i.shape)[dim]
        if i.requires_grad: _grad = True
    
    if device == 'cuda': 
          data = cuda.concatenate(tensors, dim)
    else: data =  cpu.concatenate([t.data for t in tensors], dim=dim)
    out = Tensor._new(data, shape, dtype, device, _grad, tuple(tensors))

    def _backward() -> None:
        offset = 0
        for tensor in tensors:
            if tensor.requires_grad:
                idx = []
                for i in range(out.ndim):
                    if i == dim: 
                        idx.append(slice(offset, offset + tensor.shape[dim]))
                    else: idx.append(slice(None))
                tensor.grad += out.grad[tuple(idx)]
            offset += tensor.shape[dim] 
    
    out._backward = _backward
    return out

def cat(tensors: list[Tensor], dim: int = 0) -> Tensor:
    return concatenate(tensors, dim)

def stack(tensors: list[Tensor], dim: int = 0) -> Tensor:
    tensors = [x.unsqueeze(dim) for x in tensors]
    return concatenate(tensors, dim)

def select(input: Tensor, dim: int, index: int) -> Tensor:
    return input.select(dim, index)

def unstack(input: Tensor, dim: int = 0) -> list[Tensor]:
    return input.unstack(dim)
    
def unbind(input: Tensor, dim: int = 0) -> list[Tensor]:
    return input.unbind(dim)
    
def split(
    input: Tensor, 
    split_size: int | list[int], 
    dim: int = 0
) -> list[Tensor]:
    return input.split(split_size, dim)

def chunk(input: Tensor, size: int, dim: int = 0) -> list[Tensor]:
    return input.chunk(size, dim)


