from __future__ import annotations
import builtins
from typing import Any, Self

from nectarml              import typing, cuda
from nectarml.core._tensor import tensor, Tensor, BoolTensor
from nectarml.cuda.memory  import CudaBuffer

### ABSTRACT ###

class FakeTensor(Tensor):
    def __init__(
        self:     FakeTensor,
        template: Tensor | BoolTensor,
        dtype:    typing.dtype          | None = None,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        '''Creates a new FakeTensor instance.
        
        FakeTensors are meant for functions which do not require a connection
        to the computation graph. They can be used to directly cast data to a
        different dtype or device without invoking tensor.to() and without
        introducing additional backward closures.
        
        FakeTensors can also utilize shared memory/CUDA buffers. If the input 
        dtype and device match the templates dtype and device, the FakeTensor
        will instead shadow the real Tensor to reduce memory overhead.
        
        FakeTensors cannot require grad or initiate backpropagation. They also
        cannot be cast to different DTypes or devices after creation.
        
        Args:
            template : The Tensor to serve as a template for the FakeTensor.
                       The FakeTensor's data and shape will be derived from the
                       template, as will the dtype/device if not provided.
            
            dtype    : The DType for the FakeTensor, or None to use the 
                       template Tensor's DType.
            
            device   : The device for the FakeTensor, or None to use the 
                       template Tensor's device.
            
            args     : Any additional arguments to pass to the FakeTensor.
            
            kwargs   : Any additional keyworg arguments to pass.
        '''
        device = typing.device(device) if device is not None \
            else template._device
        super().__setattr__('_device', device)
        super().__setattr__('_dtype',  dtype or template._dtype)
        
        self.data           = None
        self._buffer        = None
        
        self.shape          = template.shape
        self._requires_grad = False
        self.grad           = None
        self._prev          = set()
        self._backward      = lambda: None

        init = self._init_from_cuda \
            if template.device == 'cuda' \
          else self._init_from_cpu
        init(template)

    def _init_from_cuda(
        self:     FakeTensor,
        template: Tensor
    ) -> None:
        t   = template
        ptr = t._data_ptr
        if self.device == 'cuda':
            if self.dtype != t.dtype:
                new_ptr = (cuda.cast_ptr(ptr, self.size, t.dtype, self.dtype) 
                           if t.dtype != self.dtype 
                           else cuda.clone_ptr(ptr, self.size, self.dtype))
                self._buffer = CudaBuffer(new_ptr, self.size, self.dtype)
            else: self._buffer = t._buffer.increment()
        else:
            data      = cuda.ptr_to_cpu(ptr, t.shape, t.dtype)
            self.data = data.astype(self.dtype.numpy)

    def _init_from_cpu(
        self:     FakeTensor,
        template: Tensor
    ) -> None:
        t = template
        if self.device == 'cuda':
            data         = template.data.astype(self.dtype.numpy, copy=True)
            new_ptr      = cuda.data_to_cuda(data, self.size, self.dtype)
            self._buffer = CudaBuffer(new_ptr, self.size, self.dtype)
        else: 
            if self.dtype != t.dtype:
                  self.data = template.data.astype(self.dtype.numpy, copy=True)
            else: self.data = data
            
    def requires_grad_(self: FakeTensor, value: bool) -> Tensor:
        raise RuntimeError(
            'requires_grad_ called on FakeTensor. FakeTensors cannot be '
            'attached to the computation graph.')
        
    def _allocate_grad(self: Tensor, graph: list[tensor]) -> None:
        raise RuntimeError(
            'Tensor._allocate_grad() called on FakeTensor. FakeTensors do not '
            'support backpropagation.')
        
    def backward(self: FakeTensor, debug: bool = False) -> None:
        raise RuntimeError(
            'Tensor.backward() called on FakeTensor. FakeTensors do not '
            'support backpropagation.')
        
    def to(
        self:   FakeTensor,
        device: typing.DeviceLikeType | None = None,
        dtype:  typing.dtype | None = None
    ) -> Self: 
        raise RuntimeError(
            'Tensor.to() called on FakeTensor. FakeTensors do not support '
            'direct device or dtype changes.')

### DTYPE-SPECIFIC ###

class FloatTensor(FakeTensor):
    def __init__(
        self:     FloatTensor,
        template: Tensor,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        super().__init__(template, typing.float32, device)

class HalfTensor(FakeTensor):
    def __init__(
        self:     HalfTensor,
        template: Tensor,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        super().__init__(template, typing.float16, device)
        
class Int8Tensor(FakeTensor):
    def __init__(
        self:     Int8Tensor,
        template: Tensor,
        signed:   builtins.bool = True,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        new_dtype = typing.int8 if signed else typing.uint8
        super().__init__(template, new_dtype, device)

class Int16Tensor(FakeTensor):
    def __init__(
        self:     Int16Tensor,
        template: Tensor,
        signed:   builtins.bool = True,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        new_dtype = typing.int16 if signed else typing.uint16
        super().__init__(template, new_dtype, device)

class Int32Tensor(FakeTensor):
    def __init__(
        self:     Int32Tensor,
        template: Tensor,
        signed:   builtins.bool = True,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        new_dtype = typing.int32 if signed else typing.uint32
        super().__init__(template, new_dtype, device)
        
class Int64Tensor(FakeTensor):
    def __init__(
        self:     Int64Tensor,
        template: Tensor,
        signed:   builtins.bool = True,
        device:   typing.DeviceLikeType | None = None,
        *args:    Any,
        **kwargs: builtins.dict[builtins.str, Any]
    ) -> None:
        new_dtype = typing.int64 if signed else typing.uint64
        super().__init__(template, new_dtype, device)


