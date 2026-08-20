from nectarml.core   import tensor
from nectarml.typing import DimsType, ShapeType

def reshape(input: tensor, shape: ShapeType) -> tensor:
    '''Changes the shape of a tensor without altering its data.
    
    The reshaped output tensor will point to the same underlying data 
    (numpy.ndarray or CudaBuffer) as the input tensor. As such, changes made
    to either will be reflected in the other. 
    
    Gradients will flow unaltered from the output tensor to the input tensor
    during backpropagation if the input tensor requires grad.
    
    Args:
        input : The tensor to reshape.
        shape : The shape for the new tensor.
        
    Returns:
        tensor : The resulting reshaped tensor.
    '''
    return input.reshape(shape)

def view(input: tensor, shape: ShapeType) -> tensor:
    '''Returns a view on a given tensor with a new shape.
    
    The resulting output tensor will point to the same underlying data 
    (numpy.ndarray or CudaBuffer) as the input tensor. As such, changes made
    to either will be reflected in the other. 
    
    Gradients will flow unaltered from the output tensor to the input tensor
    during backpropagation if the input tensor requires grad.

    Args:
        input : The tensor to create a new view of.
        shape : The shape for the new tensor.
        
    Returns:
        tensor : A new tensor providing a view of the input tensor's data.
    '''
    return input.view(shape)

def flatten(input: tensor, start_dim: int = 0, end_dim: int = -1) -> tensor:
    '''Returns a view of the input tensor, flattened to a single dimension.
    
    The resulting output tensor will point to the same underlying data 
    (numpy.ndarray or CudaBuffer) as the input tensor. As such, changes made
    to either will be reflected in the other. 
    
    Gradients will flow unaltered from the output tensor to the input tensor
    during backpropagation if the input tensor requires grad.

    Args:
        input     : The tensor to flatten.
        start_dim : The first dimension to flatten.
        end_dim   : The last dimension to flatten.
        
    Returns:
        tensor : A new tensor providing a flattened view of the input tensor.
    '''
    return input.flatten(start_dim, end_dim)

def squeeze(input: tensor, dim: DimsType | None = None) -> tensor: 
    '''Removes dimensions with length one from a tensor.
    
    The resulting output tensor will point to the same underlying data 
    (numpy.ndarray or CudaBuffer) as the input tensor. As such, changes made
    to either will be reflected in the other. 
    
    Gradients will flow unaltered from the output tensor to the input tensor
    during backpropagation if the input tensor requires grad.
    
    Args:
        input : The tensor to squeeze.
        dim   : The dimension(s) to squeeze, or None to squeeze all length 1
                dimensions from the tensor.
        
    Returns:
        tensor : A new tensor providing a view of the input tensor with all
                 length 1 dimensions removed.
    '''
    return input.squeeze(dim)
    
def unsqueeze(input: tensor, dim: DimsType) -> tensor:
    '''Adds a new lenth one dimension to a given tensor's shape.
    
    The resulting output tensor will point to the same underlying data 
    (numpy.ndarray or CudaBuffer) as the input tensor. As such, changes made
    to either will be reflected in the other. 
    
    Gradients will flow unaltered from the output tensor to the input tensor
    during backpropagation if the input tensor requires grad.

    Args:
        input : The tensor to unsqueeze.
        dim   : The dimension(s) to unsqueeze along.
    
    Returns:
        tensor : A new tensor providing a view of the input tensor with the
                 newly added dimensions.
    '''
    return input.unsqueeze(dim)

def transpose(input: tensor, dim1: int, dim2: int) -> tensor:
    '''Swaps two dimensions of a given tensor and returns as a new tensor.
    
    Args:
        intput : The tensor to transpose.
        dim1   : The first dimension to transpose.
        dim2   : The second dimension to transpose.
        
    Returns:
        tensor : A new tensor object containing the input tensors data with 
                 dim1 and dim2 swapped.
    '''
    return input.transpose(dim1, dim2)

def swapdims(input: tensor, dim1: int, dim2: int) -> tensor: 
    '''Swaps two dimensions of a given tensor and returns as a new tensor.
    
    Args:
        intput : The tensor to transpose.
        dim1   : The first dimension to transpose.
        dim2   : The second dimension to transpose.
        
    Returns:
        tensor : A new tensor object containing the input tensors data with 
                 dim1 and dim2 swapped.
    '''
    return input.transpose(dim1, dim2)

def permute(input: tensor, dims: DimsType | None) -> tensor:
    '''Rearranges a given tensors dimensions by a given order.
    
    Args:
        input : The tensor to permute.
        dims  : A tuple of ints defining the order to rearrange the tensor's
                dimensions in.
        
    Returns:
        tensor : A new tensor object containing the input tensor's data with 
                 the permuted shape.
                 
    Examples:
        ```  
        x = nectarml.rand((3, 256, 256))
        y = F.permute(input, (1, 2, 0))
        print(y.shape)
        ```
        Result: `nectarml.Size(256, 256, 3)`
    '''
    return input.permute(dims)

def expand(input: tensor, shape: ShapeType) -> tensor:
    return input.expand(shape)

def broadcast_to(input: tensor, shape: ShapeType) -> tensor:
    return input.expand(shape)

def unfold(input: tensor, dimension: int, size: int, step: int) -> tensor:
    return input.unfold(dimension, size, step)

def flip(input: tensor, dims: DimsType) -> tensor:
    '''Reverses tensor elements along given dimension(s).
    
    Args:
        input : The tensor to flip.
        dims  : An integer, or list of integers, denoting the dimensions(s) to 
                reverse the tensor along.
                
    Returns:
        tensor : The resulting tensor from the flip operation.
    '''
    return input.flip(dims)
