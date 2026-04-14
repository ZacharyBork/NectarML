from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tensor import Tensor
    
import builtins
from typing import Literal, Self

import numpy as np

from nectarml import typing, cpu, cuda
from nectarml.tensor._tensor import tensor
from nectarml.cuda.memory import CudaBuffer
   
class BoolTensor(tensor):    
    def __init__(
        self:          Tensor,
        data:          typing.ArrayLike,
        shape:         typing.Size | tuple[builtins.int, ...] | None = None,
        dtype:         typing.DTypeLike = typing.bool_,
        device:        Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children:     tuple[tensor, ...] = ()
    ) -> None:
        assert dtype == typing.bool_,  'Boolean tensors must have bool_ DType.'
        assert requires_grad == False, 'Boolean tensors cannot require grad.'
        assert _children == (),        'Boolean tensors cannot have _children.'
        
        if not isinstance(data, np.ndarray): data = np.array(data)
        shape = shape if isinstance(shape, typing.Size) else \
                typing.Size(shape or data.shape)
        
        super().__init__(
            data=BoolTensor._build_data(data, shape, device), 
            shape=shape, dtype=typing.bool_, 
            device=device, requires_grad=False)

    ### INIT ###
    
    @classmethod
    def _build_data(
        cls:    type[Self],
        data:   np.ndarray,
        shape:  typing.Size,
        device: Literal['cpu', 'cuda'] 
    ) -> np.ndarray | CudaBuffer:
        assert data.dtype == np.bool_, (
            f'BoolTensor.__init__() expecting bool_ type data, but recieved '
            f'data of type {data.dtype}.')
        
        match device:
            case 'cpu': ref = data
            case 'cuda': 
                ref = CudaBuffer(
                    cuda.data_to_cuda(data, shape.numel(), np.bool_), np.bool_)
            case _: raise ValueError(f'Invalid device type: {device}')
        return ref
    
    ### CASTING ###
    
    def to(
        self:   BoolTensor,
        device: Literal['cpu', 'cuda'] | None = None,
        dtype:  typing.DTypeLike | None = None
    ) -> BoolTensor | Tensor: 
        '''Casts BoolTensor to new device and/or Dtype.
        
        If both device and DType are the same as the device and DType of the
        tensor this is called on, this method will return a reference to the
        original tensor object. If you would like to make a duplicate of a
        given tensor, please see tensor.clone() instead.
        
        NOTE: The returned Tensor will have requires_grad=False by default.
        
        Args:
            device : The device to cast the tensor to ["cpu", "cuda"].
            dtype : The Dtype to cast the tensor to.
            
        Returns:
            tensor : The resulting tensor from the cast operation.
        '''
        device = device or self.device
        dtype  = dtype  or self.dtype
        if device == self.device and dtype == self.dtype: return self
        if dtype != typing.bool_: return self._to_numerical().to(device, dtype)
        else: return self

    ### COMPARISON ###
    
    def __eq__(
        self:  BoolTensor, 
        other: BoolTensor | bool
    ) -> BoolTensor:
        '''tensor elementwise equality operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is equal
                to the corresponding value in the "other" tensor. Note: This
                return tensor can be turned into a binary mask like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x == y).to(x.device, x.dtype)
        '''        
        if isinstance(other, BoolTensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda': 
            data = cuda.math.equal(self, other, out_shape)
        else: data = cpu.math.equal(self, other)
        return BoolTensor._new(data, out_shape, typing.bool_, self.device)
