from nectarml.core         import Tensor
from nectarml.cuda         import combination as cuda
from nectarml.cpu          import combination as cpu
from nectarml.amp.autocast import autocast_state
from nectarml.amp.utils    import get_promotion_dtype

def concatenate(tensors: list[Tensor], dim: int = 0) -> Tensor:
    '''Concatenates tensors along a given dimension and returns the result.
    
    Example:
    
        tensors = [Tensor((1, 3, 256, 256)), Tensor((1, 3, 256, 256))]
        x = F.concatenate(tensors, dim=0)
        print(x.shape)
        ------------------------------------------------------------------
        Result: nectarml.Size(2, 3, 256, 256) 

    All input tensors must be on the same device, and, if autocast('cuda') is
    no enabled, must also have the same DType. If autocast('cuda') is enabled,
    all input tensors will be cast to the highest ranking DType (i.e.
    fp16->fp32) of all of the input tensors if necessary.
    
    Args:
        tensors : A list of tensors to concatenate.
        dim     : The dimension along which to concatenate the input tensors.
        
    Returns:
        Tensor : The resulting tensor from the concatenation.
    '''
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
    '''Concatenates tensors along a given dimension and returns the result.
    
    Example:
    
        tensors = [Tensor((1, 3, 256, 256)), Tensor((1, 3, 256, 256))]
        x = F.cat(tensors, dim=0)
        print(x.shape)
        
        Result: nectarml.Size(2, 3, 256, 256) 
    
    All input tensors must be on the same device, and, if autocast('cuda') is
    no enabled, must also have the same DType. If autocast('cuda') is enabled,
    all input tensors will be cast to the highest ranking DType (i.e.
    fp16->fp32) of all of the input tensors if necessary.
    
    Args:
        tensors : A list of tensors to concatenate.
        dim     : The dimension along which to concatenate the input tensors.
        
    Returns:
        Tensor : The resulting tensor from the concatenation.
    '''
    return concatenate(tensors, dim)

def stack(tensors: list[Tensor], dim: int = 0) -> Tensor:
    '''Stacks input tensors along a new dimension.
    
    Example:
    
        tensors = [Tensor((3, 32, 32)), Tensor((3, 32, 32))]
        x = F.stack(tensors, dim=0)
        print(x.shape)
        ------------------------------------------------------------------
        Result: nectarml.Size(2, 3, 32, 32) 

    All input tensors must be on the same device, and, if autocast('cuda') is
    no enabled, must also have the same DType. If autocast('cuda') is enabled,
    all input tensors will be cast to the highest ranking DType (i.e.
    fp16->fp32) of all of the input tensors if necessary.
    
    Args:
        tensors : A list of tensors to stack.
        dim     : The dimension along which to concatenate the input tensors.
        
    Returns:
        Tensor : The resulting tensor from the concatenation.
    '''
    tensors = [x.unsqueeze(dim) for x in tensors]
    return concatenate(tensors, dim)

def select(
    input:   Tensor, 
    dim:     int, 
    index:   int,
    keepdim: bool = False
) -> Tensor:
    return input.select(dim, index, keepdim)

def unstack(
    input:    Tensor, 
    dim:      int = 0, 
    keepdim: bool = False
) -> list[Tensor]:
    '''Unstacks input tensor along a given dimension dimension.
    
    See: nectarml.functional.unbind() for more info.
    
    Args:
        input   : The tensor to unstack.
        dim     : The dimension along which unstack the input tensor.
        keepdim : If true, the resulting tensors will be unsqueezed along the
                  dimension on which they were unbound. See example 2.
        
    Returns:
        list[Tensor] : The tensors resulting from the unbind operation.
    '''
    return input.unstack(dim, keepdim)
    
def unbind(
    input:    Tensor, 
    dim:      int = 0, 
    keepdim: bool = False
) -> list[Tensor]:
    '''Unbinds input tensor along a given dimension dimension.
    
    Example 1:
    
        x = Tensor((2, 3, 256, 256))
        y: list[Tensor] = F.unbind(x, dim=0)
        print([t.shape for t in y])
        ------------------------------------------------------------------
        Result: [nectarml.Size(3, 256, 256), nectarml.Size(3, 256, 256)]
        
    Example 2:
        
        x = Tensor((2, 3, 32, 32))
        y: list[Tensor] = F.unbind(x, dim=0, keepdim=True)
        print([t.shape for t in y])
        ------------------------------------------------------------------
        Result: [nectarml.Size(1, 3, 32, 32), nectarml.Size(1, 3, 32, 32)]
    
    Args:
        input   : The tensor to unbind.
        dim     : The dimension along which unbind the input tensor.
        keepdim : If true, the resulting tensors will be unsqueezed along the
                  dimension on which they were unbound. See example 2.
        
    Returns:
        list[Tensor] : The tensors resulting from the unbind operation.
    '''
    return input.unbind(dim, keepdim)
    
def split(
    input:      Tensor, 
    split_size: int | list[int], 
    dim:        int = 0
) -> list[Tensor]:
    return input.split(split_size, dim)

def chunk(input: Tensor, size: int, dim: int = 0) -> list[Tensor]:
    return input.chunk(size, dim)


