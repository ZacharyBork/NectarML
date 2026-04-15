from __future__ import annotations

import types
import builtins
from typing import Any, Literal, Self
from collections.abc import Callable

import numpy as np

from nectarml import typing, cpu, cuda, autograd
from nectarml.cuda.memory import CudaBuffer

class tensor:
    _class_type_nectar_tensor  = True
    _subclasses: builtins.dict = {}
    
    def __init__(
        self:          tensor,
        data:          np.ndarray | CudaBuffer,
        shape:         typing.Size | tuple[builtins.int, ...],
        dtype:         typing.DTypeLike,
        device:        Literal['cpu', 'cuda'],
        requires_grad: bool = False,
        _children:     tuple[tensor, ...] = ()
    ) -> None:
        '''Initializes a new tensor object.
        
        Args:
            data : Either an np.ndarray object of tensor data, or a CudaBuffer
                pointing to the tensor's data in CUDA memory.
            shape : A tuple[int, ...] or Size object defining the tensor's
                shape.
            dtype : A DtypeLike defining the data type for the new tensor.
            device : The device for the new tensor, "cpu" or "cuda".
            requires_grad : A boolean defining whether the new tensor should
                require grad or not. If True, a grad tensor will be created
                and assigned to the new tensor, and the new tensor will be
                included in the computational graph to participate in gradient
                backpropagation. If False, the tensor will not be included in
                the computational graph, and will not contribute to the 
                network's gradients.
            _children : The _prev tensors for the newly created tensor. If the
                new tensor is included in the computational graph, the 
                gradients from the new tensor will flow back to the _children
                tensors during backpropagation. Used for autograd operations,
                generally not set manually.
        '''
        self.shape          = shape
        self._dtype         = dtype
        self.device         = device
        self._requires_grad = requires_grad
        
        self.grad:             tensor | None = None
        self.data:         np.ndarray | None = None
        self._buffer:      CudaBuffer | None = None
        self._device_id: builtins.int | None = None
        
        self._backward: Callable = lambda : None
        self._prev:  set[tensor] = set(_children)
        
        self._init_tensor(data)         
        
    ### INIT ###
        
    def _init_tensor(
        self: tensor, 
        data: np.ndarray | CudaBuffer
    ) -> None:
        '''Initializes a tensor from given data and optional shape.
        
        Args:
            data : Either an np.ndarray object of tensor data, or a CudaBuffer
                pointing to the tensor's data in CUDA memory.
                
        Raises:
            ValueError : If data is uintptr to CUDA tensor data and shape is
                not provided.
            ValueError : If tensor's device type is not valid (i.e. not "cpu"
                or "cuda").
        '''
        if self.device == 'cpu':
            assert isinstance(data, np.ndarray), \
                f'Invalid data type for CPU tensor: {type(data)}'
            self.data = data
        elif self.device == 'cuda': 
            assert isinstance(data, CudaBuffer), \
                f'Invalid data type for CUDA tensor: {type(data)}'
            self._device_id = 0 # NEEDS TO BE UPDATED FOR REAL MULTI-GPU ID
            self._buffer = data
        else: raise ValueError(f'Invalid device type: {self.device}')
    
    @classmethod
    def _new(
        cls:    type[Self],
        data:   builtins.int | np.ndarray,
        shape:  typing.Size | tuple[builtins.int, ...],
        dtype:  typing.DTypeLike,
        device: Literal['cpu', 'cuda'],
        requires_grad: bool = False,
        _children: tuple = ()
    ) -> Self:
        out = cls.__new__(cls)
        out.device         = device
        out._device_id     = 0
        out._dtype         = dtype
        out.shape          = shape if isinstance(shape, typing.Size) \
                                else typing.Size(shape)
        
        out._requires_grad = requires_grad
        out.grad           = None
        out._prev          = set(_children)
        out._backward      = lambda: None
        
        if device == 'cuda': 
            out._buffer = CudaBuffer(data, dtype)
            out.data = None
        elif device == 'cpu':
            out._buffer = None
            out.data = data

        return out
    
    @classmethod
    def _from_data(
        cls:   type[Self],
        data:  np.ndarray, 
        shape: typing.Size | tuple[builtins.int, ...], 
        dtype: typing.DTypeLike, 
        requires_grad: bool = False,
        _children: tuple = ()
    ) -> Self:
        '''Helper method to duplicate CPU tensors which share underlying data.
        
        Args:
            cls : tensor class type.
            data : The data reference to assign to the new tensor.
            shape : The shape of the new tensor.
            dtype : The Dtype of the new tensor.
            requires_grad : Whether the new tensor should require grad.
            _children : The backprop children for the new tensor
            
        Return:
            tensor : The newly created tensor.
        '''
        out = cls.__new__(cls)
        out.device         = 'cpu'
        out._device_id     = 0
        out._dtype         = dtype
        out.shape          = shape
        
        
        out.data           = data
        out._buffer        = None
        
        out._requires_grad = requires_grad
        out.grad           = None
        out._prev          = set(_children)
        out._backward      = lambda: None
        
        return out
    
    @classmethod
    def _from_buffer(
        cls:    type[Self],
        buffer: CudaBuffer,
        shape:  typing.Size,
        dtype:  typing.DTypeLike,
        requires_grad: bool = False,
        _children: tuple = ()
    ) -> Self:
        '''Helper method to duplicate CUDA tensors which share underlying data.
        
        Args:
            cls : tensor class type.
            buffer : The CudaBuffer reference to assign to the new tensor.
            shape : The shape of the new tensor.
            dtype : The Dtype of the new tensor.
            requires_grad : Whether the new tensor should require grad.
            _children : The backprop children for the new tensor
            
        Return:
            tensor : The newly created tensor.
        '''
        out = cls.__new__(cls)
        out.device         = 'cuda'
        out._device_id     = 0
        out._dtype         = dtype
        out.shape          = shape if isinstance(shape, typing.Size) \
                                else typing.Size(shape)
        out.data           = None
        out._buffer        = buffer.increment()
        
        out._requires_grad = requires_grad
        out.grad           = None
        out._prev          = set(_children)
        out._backward      = lambda: None
        
        return out
    
    @classmethod
    def _temporary(
        cls:          type[Self],
        input:        Self,
        data_ptr:     builtins.int,
        target_dtype: typing.DTypeLike
    ) -> Self:
        assert input.device == 'cuda', \
            'tensor._reference is only valid for CUDA tensors.'
        
        tmp = cls.__new__(cls)
        tmp.device         = input.device
        tmp._device_id     = input._device_id
        tmp._dtype         = target_dtype
        tmp.shape          = input.shape
        
        tmp._buffer        = CudaBuffer(data_ptr, target_dtype)
        
        tmp._requires_grad = False
        tmp.grad           = None
        tmp._prev          = set()
        tmp._backward      = lambda: None

        return tmp
    
    ### PROPERTIES ###
      
    @property
    def _data_ptr(self: Self) -> builtins.int | None:
        '''Property access for CUDA pointer to tensor's data.
        
        Returns:
            int | None : If the given tensor owns a CudaBuffer (i.e. if its
                device is "cuda"), returns the uintptr to the tensor's data in
                CUDA memory. Otherwise returns NoneType.
        '''
        if self._buffer is not None: return self._buffer.ptr
        return None
      
    @property
    def dtype(self: Self) -> typing.DTypeLike:
        '''Property access for tensor's Dtype.
        
        Returns:
            typing.DtypeLike : The Dtype of the tensor.
        '''
        return self._dtype
    
    @property
    def ndim(self: tensor) -> builtins.int:
        '''Property access for number of dims in tensor.
        
        Returns:
            int : The number of dimensions in the given tensor's shape.
        '''
        return self.shape.ndim
    
    @property
    def size(self: Self) -> builtins.int:
        '''Propert access for tensor size (i.e. numel).
        
        Returns:
            int : The number of elements in the given number shape. Equivelent
                to math.prod(tensor.shape).
        '''
        return self.shape.numel()
           
    @property
    def is_contiguous(self: Self) -> bool:
        if self.device == 'cuda': return True
        else: return self.data.flags['C_CONTIGUOUS']
        
    @property
    def requires_grad(self: tensor) -> bool:
        '''Property access for tensor's requires_grad value.
    
        Returns:
            bool : True if given tensor requires grad, otherwise False.
        '''
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self: Self, value: bool) -> None:
        '''Setter for tensor's requires_grad value.
        
        This setter will allocate a grad tensor for the given data tensor if
        value=True and the tensor's grad is None. If value=False, it will set
        the given tensor's grad to None.
        
        Args:
            value : True to enable grad on the given tensor, False to disable.
        '''
        if self.dtype != typing.bool_: 
            self._requires_grad = value and autograd.is_grad_enabled()
        else: self._requires_grad = False
        
    @property
    def _mod(self: Self) -> types.ModuleType:
        return cuda if self.device == 'cuda' else cpu
        
    ### CLASS METHODS ###
    
    @classmethod
    def _normalize_dim(
        cls:  type[Self],
        dims: builtins.int | tuple[builtins.int, ...] | None, 
        ndim: builtins.int
    ) -> builtins.int | tuple[builtins.int, ...] | None:
        if dims is None: return dims
        
        if isinstance(dims, builtins.int):
            if dims < 0: dims = ndim + dims
            if dims < 0 or dims >= ndim:
                raise ValueError(
                    f'dim {dims} out of range for tensor with {ndim} dims')
            return dims
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
              dims = tuple(dims[0])
        else: dims = tuple(dims)
        return tuple(d % ndim for d in dims)
            
    @classmethod
    def _broadcast_shape(
        cls:     type[Self],
        a_shape: tuple[builtins.int, ...] | typing.Size, 
        b_shape: tuple[builtins.int, ...] | typing.Size
    ) -> tuple[builtins.int, ...]:
        '''Builds output shape for ops between tensors with different shapes.
        
        Args:
            a_shape : The shape of the first tensor (generally self).
            b_shape : The shape of the other tensor in the operation.
            
        returns:
            tuple[int, ...] : The output shape for the resulting tensor from 
                the given operation.
        '''
        ndim = max(len(a_shape), len(b_shape))
        a_padded = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
        b_padded = (1,) * (ndim - len(b_shape)) + tuple(b_shape)
        
        out_shape = []
        for a, b in zip(a_padded, b_padded):
            if a == b:   out_shape.append(a)
            elif a == 1: out_shape.append(b)
            elif b == 1: out_shape.append(a)
            else:
                raise ValueError(
                    f'Shapes {a_shape} and {b_shape} are not broadcastable')
        
        return typing.Size(tuple(out_shape))
        
    @classmethod
    def _normalize_shape_input(
        csl: type[Self],
        *shape: int 
              | tuple[builtins.int, ...]
              | typing.Size
    ) -> typing.Size:
        if len(shape) == 1 \
        and isinstance(shape[0], (tuple, list, typing.Size)):
              return typing.Size(list(shape[0]))
        else: return typing.Size(shape)
            
    ### DEVICE / DTYPE ###
        
    def to(
        self:   tensor,
        device: Literal['cpu', 'cuda'] | None = None,
        dtype:  typing.DTypeLike | None = None
    ) -> Self: 
        '''Casts tensor to new device and/or Dtype.
        
        If both device and DType are the same as the device and DType of the
        tensor this is called on, this method will return a reference to the
        original tensor object. If you would like to make a duplicate of a
        given tensor, please see tensor.clone() instead.
        
        Args:
            device : The device to cast the tensor to ["cpu", "cuda"].
            dtype : The Dtype to cast the tensor to.
            
        Returns:
            tensor : The resulting tensor from the cast operation.
        '''        
        device = device or self.device
        dtype  = dtype  or self.dtype
        if device == self.device and dtype == self.dtype: return self

        if device == 'cuda':
            if self.device == 'cpu':
                src = self if self.is_contiguous else self.contiguous()
                tmp_ptr = cuda.to_cuda(src)
                if dtype != self.dtype:
                    data = cuda.cast_tensor_by_reference(
                        tmp_ptr, self.size, self.dtype, dtype)
                    cuda.free_cuda(tmp_ptr)
                else: data = tmp_ptr
            else: data = cuda.cast_tensor(self, dtype)
            shape = self.shape
                    
        elif device == 'cpu':
            if self.device == 'cpu': data = self.data.astype(dtype)
            else: data = cuda.to_cpu(self, dtype)
            shape = typing.Size(data.shape)
        else: raise ValueError(f'Invalid device type: {device}')
        
        new = self._new(data, shape, dtype, device, self.requires_grad)
        new.grad = self.grad.to(device) if self.grad is not None else None

        if self.requires_grad:
            original_device = self.device
            
            def _backward() -> None:
                if new.grad is None: return
                grad_f32 = new.grad.to(original_device, typing.float32)
                if self.grad is None: self.grad = grad_f32
                else: self.grad += grad_f32

            new._prev     = {self}
            new._backward = _backward

        return new
            
    def cuda(self: tensor) -> Self: 
        '''Convenience function to cast given tensor to CUDA device.
        
        When called on a tensor who's device is already "cuda", this method 
        will return a reference to the original tensor object. If you would 
        like to make a duplicate of a given tensor, please see tensor.clone().
        
        Returns:
            tensor : The resulting CUDA tensor from the cast operation.
        '''
        return self.to(device='cuda')

    def cpu(self: tensor) -> Self: 
        '''Convenience function to cast given tensor to CPU device.
        
        When called on a tensor who's device is already "cpu", this method 
        will return a reference to the original tensor object. If you would 
        like to make a duplicate of a given tensor, please see tensor.clone().
        
        Returns:
            tensor : The resulting CPU tensor from the cast operation.
        '''
        return self.to(device='cpu')
            
    ### DATA UTILS ###
    
    def numpy(self: tensor) -> np.ndarray:
        '''Returns the tensor's data as a numpy.ndarray.
        
        Returns:
            np.ndarray : The tensor's data as a numpy.ndarray.
        '''
        if self.device == 'cuda': 
            return cuda.to_cpu(self, self.dtype)
        return self.data
    
    def tolist(self: tensor) -> list[Any]:
        '''Returns the tensor's data as nested list.
        
        Returns:
            list[Any] : The tensor's data as a nested list.
        '''
        return self.numpy().tolist()
    
    def item(self: tensor) -> builtins.int | builtins.float:
        '''Returns the value of the given tensor as a float or int.
        
        Returns:
            int | float : The value of the tensor.
            
        Raises:
            RuntimeError : If called on tensor with more than a single element.
        '''
        if self.numel() != 1:
            raise RuntimeError(
                'tensor.item() can only be called on tensors with 1 element.')
        return self.numpy().item()
    
    def is_floating_point(self: tensor) -> bool:
        '''tensor floating point Dtype check.
        
        Returns:
            bool : True if tensor's Dtype is float, float16, or float32,
                otherwise False
        '''
        return self.dtype in [typing.float, typing.float16, typing.float32]
    
    def is_cuda(self: tensor) -> bool:
        '''CUDA device check. Equivalent to tensor.device == 'cuda'.
        
        Returns:
            bool : True if the tensor's device is 'cuda', otherwise False.
        '''
        return self.device == 'cuda'
    
    def is_cpu(self: tensor) -> bool:
        '''CPU device check. Equivalent to tensor.device == 'cpu'.
        
        Returns:
            bool : True if the tensor's device is 'cpu', otherwise False.
        '''
        return self.device == 'cpu'
    
    def dim(self: tensor) -> builtins.int:
        '''Returns ndim of tensor.shape. Equivalent to tensor.ndim
        
        Returns:
            int : The number of dimensions in the given tensor's shape.
        '''
        return self.shape.ndim
    
    def numel(self: tensor) -> builtins.int:
        '''Returns the number of elements in tensor's shape.
        
        Functional equivalent of math.prod(tensor.shape).
        
        Returns:
            int : The number of elements in the given tensor's shape.
        '''
        return self.shape.numel()
    
    ### UTILS ###
    
    def contiguous(self: tensor) -> Self:
        '''Creates new tensor from original with contiguous memory layout.
        
        If the original tensor's data is already contiguous, this function will
        instead return a reference to the original tensor. This functional is
        differentiable; gradients will flow back from the new tensor to the
        original if the original is included in the computation graph.
        
        Returns:
            tensor : Either a new tensor with the original tensor's data in a
                contiguous memory layout, or a reference to the original tensor
                if the original tensor was already contiguous when this method
                was called.
        '''
        if self.is_contiguous: return self
        if self.device == 'cuda': data = cuda.clone(self)
        else: data = np.ascontiguousarray(self.data)
        
        out = self._new(data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        self_requires_grad = self.requires_grad
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
        
        out._backward = _backward
        return out
    
    def clone(self: tensor) -> Self:
        '''Creates and returns a clone of the tensor.
        
        This method is differentiable. The new tensor's gradients will flow
        back from the newly created tensor to the tensor this method was called
        on. If you would like to avoid this, please see tensor.detach().
        
        Returns:
            tensor : The newly created clone tensor.
        '''
        if self.device == 'cuda': data = cuda.clone(self)
        else: data = self.data.copy()
        
        out = self._new(data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        self_requires_grad = self.requires_grad
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
        
        out._backward = _backward
        return out
    
    def copy_(self: tensor, other: tensor) -> Self:
        '''Copies data from other tensor to this tensor in-place.
        
        Requires that other tensor have the same shape and dtype as the tensor
        this method is called from. The tensors can be on different devices.
        The tensor calling this method will remain on whatever device it 
        started on regardless.
        
        Args:
            other : The tensor to copy the data from.
            
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        assert self.shape == other.shape, \
            f'copy_ requires tensors to have the same shape.'
        assert self.dtype == other.dtype, \
            f'copy_ requires tensors to have the same dtype.'
        if self.device == 'cuda':
            if other.device == 'cuda': copy_ptr = cuda.utils.clone(other)
            else: copy_ptr = cuda.data_to_cuda(
                other.data, other.size, other.dtype)
            old_buffer = self._buffer
            self._buffer = CudaBuffer(copy_ptr, self.dtype)
            old_buffer.decrement()
        else: np.copyto(self.data, other.numpy())
        return self
    
    def cuda_build_shape_(self: tensor) -> None:
        '''Builds CUDA tensor's shape in-place from tensor's actual data.
        
        This method will call tensor.numpy() internally on the given
        tensor, temporarily copying the tensor's data from VRAM to system
        memory. It will then overwrite the given tensors shape in-place with
        the shape of the actual data. 
        
        NOTE: This is not very performance friendly, especially on larger 
        tensors. Generally only use this for debugging purposes.
        '''
        self.shape = typing.Size(self.numpy().shape)
    
    def _validate_other(self: tensor, other: tensor) -> None:
        assert isinstance(other, tensor)
        assert self.device == other.device, (
            f'Expected all tensors to be on the same device, but found at '
            f'least two devices, {self.device} and {other.device}.')

    ### GETTERS / SETTERS ###

    def __getitem__(
        self: tensor, 
        idx:  tensor 
           |  tuple[tensor, tensor]
           |  builtins.int
           |  slice
           |  tuple[slice]
    ) -> Self:
        self_requires_grad = self.requires_grad
        if not isinstance(idx, tuple): idx = (idx,)
        tensor_indices = [
            (i, v) for i, v in enumerate(idx) if isinstance(v, tensor)]
        for _, index in tensor_indices:
            assert index.device == self.device, (
                    f'tensor.__getitem__() expects input tensor and index '
                    f'tensor to be on same device, but found two devices: '
                    f'{self.device} and {index.device}')
            assert index.dtype in (typing.int32, typing.int64), \
                'tensor index must be integer dtype'
    
        if len(tensor_indices) == 0:
            idx = list(idx)
            while len(idx) < self.ndim: idx.append(slice(None))
            
            squeeze_dims = []
            normalized: list[slice] = []
            for i, index in enumerate(idx):
                if isinstance(index, builtins.int):
                    if index < 0: index = self.shape[i] + index
                    normalized.append(slice(index, index + 1, 1))
                    squeeze_dims.append(i)
                elif isinstance(index, slice): normalized.append(index)
                else: raise ValueError(f'Index type not valid: {type(index)}')
            
            starts, stops, steps, counts = [], [], [], []
            for i, s in enumerate(normalized):
                start, stop, step = s.indices(self.shape[i])
                count = len(range(start, stop, step))
                starts.append(start)
                stops.append(stop)
                steps.append(step)
                counts.append(count)

            if self.device == 'cuda':
                out_data = cuda.indexing.slice_tensor(
                    self, starts, counts, steps)
            else: out_data = self.data[tuple(normalized)]

            out_shape = typing.Size(counts)
            out = self._new(out_data, out_shape, self.dtype, self.device, 
                self.requires_grad, _children=(self,))
            for dim in reversed(squeeze_dims): out = out.squeeze(dim)
            
            def _backward() -> None:
                if self_requires_grad:
                    grad = self._new(
                        np.zeros(self.shape, typing.float32), 
                        self.shape, typing.float32, self.device, 
                        requires_grad=False)
                    grad[tuple(normalized)] = out.grad
                    self.grad += grad
            
            out._backward = _backward
            return out
            
        elif len(tensor_indices) == 1:
            dim, t_idx = tensor_indices[0]
            
            result = self
            for i, v in enumerate(idx):
                if isinstance(v, slice) and v != slice(None):
                    result = result[tuple(
                        v if j == i else slice(None) 
                        for j in range(result.ndim))]
            
            return result.index_select(dim, t_idx.flatten()).reshape(
                t_idx.shape + result.shape[dim+1:])
                
        elif len(tensor_indices) == 2:
            dim0, idx0 = tensor_indices[0]
            dim1, idx1 = tensor_indices[1]
            
            assert idx0.shape == idx1.shape, \
                'Multiple tensor indices must have the same shape'
            assert dim0 == 0 and dim1 == 1, (
                'Paired tensor indexing only supported on dims '
                '0 and 1 currently')
            
            K     = idx0.numel()
            flat0 = idx0.flatten()
            flat1 = idx1.flatten()
            
            if self.ndim == 2:
                rows   = self.index_select(0, flat0)
                result = rows.gather(1, flat1.unsqueeze(1))
                result = result.squeeze(1)
                return result.reshape(idx0.shape)
            else:
                rows      = self.index_select(0, flat0)
                remaining = self.shape[2:]
                
                flat1_exp = flat1.reshape((K,) + (1,) * (self.ndim - 1))
                flat1_exp = flat1_exp.expand((K, 1) + remaining)
                result    = rows.gather(1, flat1_exp).squeeze(1)
                return result.reshape(idx0.shape + remaining)
        
        else:
            raise NotImplementedError(
                f'tensor indexing with {len(tensor_indices)} tensor indices '
                f'is not currently supported. Use gather() instead.')

    def __setitem__(
        self:  tensor, 
        idx:   builtins.int | slice | tuple[slice],
        value: Any
    ) -> None:
        '''
        NOTE: tensor.__setitem__() modifies the given tensor in-place. This
        operation does not support autograd. Calling this method on a tensor in 
        the autograd graph will corrupt the gradients of any tensor which 
        depends on it.'
        '''
        if not isinstance(idx, tuple): idx = (idx,)
        idx = list(idx)
        while len(idx) < self.ndim: idx.append(slice(None))
        
        normalized = []
        for i, index in enumerate(idx):
            if isinstance(index, builtins.int):
                if index < 0: index = self.shape[i] + index
                normalized.append(slice(index, index + 1, 1))
            elif isinstance(index, slice): normalized.append(index)
            else: raise ValueError(f'Index type not valid: {type(index)}')
        
        starts, counts, steps = [], [], []
        for i, s in enumerate(normalized):
            start, stop, step = s.indices(self.shape[i])
            count = len(range(start, stop, step))
            starts.append(start)
            counts.append(count)
            steps.append(step)
        
        if self.device == 'cuda':
            if not isinstance(value, tensor):
                value = self._new(
                    np.full(counts, value, dtype=self.dtype),
                    counts, self.dtype, self.device)
            new_ptr = cuda.indexing.index_put(
                self, starts, counts, steps, value)
            old_buffer = self._buffer
            self._buffer = CudaBuffer(new_ptr, self.dtype)
            old_buffer.decrement()
        else:
            if isinstance(value, tensor): value = value.data
            self.data[tuple(normalized)] = value
    
    ### INSPECTION ###
        
    def __str__(self: tensor) -> str: 
        '''Returns tensor info (data, device) as a formatted string.
        
        Returns:
            str : The tensor info string.
        '''
        data = self.numpy()
        if data.size == 1: 
            data = data.tolist()
            if isinstance(data, list): data = data[0]
            data_str = str(data)
        else: data_str = np.array2string(data, separator=', ', precision=4)
        
        class_name = self.__class__.__name__
        data_str = data_str.replace('\n', '\n' + ' ' * (len(class_name) + 1))
        device_str = f'{self.device}' 
        if self.device == 'cuda' and self._device_id is not None:
            device_str = f'{device_str}:{self._device_id}'
        return f'{class_name}({data_str}, device=\'{device_str}\')'
    
    def __repr__(self: tensor) -> str:
        data = self.numpy()
        if data.size == 1: 
            data = data.tolist()
            if isinstance(data, list): data = data[0]
            data_str = str(data)
        else: data_str = np.array2string(data, separator=', ', precision=4)
        return (
            f'{self.__class__.__name__}: [\n'
            f'    shape: {self.shape},\n'
            f'    dtype: {self.dtype}\n'
            f'    requires_grad: {self.requires_grad},\n'
            f'    data: {data_str},\n'
            f'    _prev: {self._prev}\n'
            f']'
        )
    
    def __len__(self: tensor) -> builtins.int: 
        '''Gets the length (in elements) of the tensor's data.
        
        Returns:
            int : The tensor's length (in elements).
        '''
        return self.size
    
    def __hash__(self: tensor) -> builtins.int: 
        '''Hash override. Returns memory address of tensor.
        
        Returns:
            int : Address in system memory of the tensor.
        '''
        return id(self)
    
    ### GARBAGE COLLECTION ###
    
    def __del__(self: tensor) -> None:
        '''tensor garbage collection.
        
        Used to decrement the CudaBuffer of the tensor if device="cuda". If
        CudaBuffer reference count reaches zero when this tensor is deleted,
        the CUDA memory of the tensor is cleared and the garbage collector
        is then allowed to clean up the CudaBuffer object.
        '''
        if self.device == 'cuda' and self._buffer is not None:
            self._buffer.decrement()

    ### RESHAPING ###
    
    def reshape(
        self:   tensor, 
        *shape: builtins.int | tuple[builtins.int, ...] | typing.Size 
    ) -> Self:
        shape = tensor._normalize_shape_input(*shape)
        if -1 in shape:
            shape = list(shape)
            idx   = shape.index(-1)
            known = 1
            for i, s in enumerate(shape):
                if i != idx: known *= s
            shape[idx] = self.size // known
            shape = typing.Size(shape)
        
        self_requires_grad = self.requires_grad
        orig_shape         = self.shape
        
        if self.device == 'cuda':
            out = self._from_buffer(
                self._buffer, shape, self.dtype, self.requires_grad,
                _children=(self,))
        else:
            out = self._new(
                cpu.shapes.reshape(self.data, shape), shape, self.dtype, 
                self.device, self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad.reshape(orig_shape)
                        
        out._backward = _backward
        return out
    
    def view(
        self:   tensor,
        *shape: builtins.int | tuple[builtins.int, ...] | typing.Size 
    ) -> Self:
        shape = tensor._normalize_shape_input(*shape)
        
        total = self.numel()
        if -1 in shape:
            idx = shape.index(-1)
            known = 1
            for i, s in enumerate(shape):
                if i != idx: known *= s
            shape = shape[:idx] + (total // known,) + shape[idx+1:]
        
        assert self.is_contiguous, \
            'view() called on non-contiguous tensor. Use reshape() instead.'
        assert np.prod(shape) == total, \
            f'view() cannot change number of elements: {self.shape} -> {shape}'
        
        self_requires_grad = self.requires_grad
        orig_shape = self.shape
        
        if self.device == 'cuda':
            out = self._from_buffer(
                self._buffer, shape, self.dtype, self.requires_grad,
                _children=(self,))
        else:
            out = self._new(
                self.data.reshape(shape), shape, self.dtype, 
                self.device, self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad.view(orig_shape)
        
        out._backward = _backward
        return out
        
    def flatten(
        self:      tensor, 
        start_dim: builtins.int = 0,
        end_dim:   builtins.int = -1
    ) -> Self:
        end_dim = end_dim if end_dim >= 0 else self.ndim + end_dim
        new_shape = (
            self.shape[:start_dim]
          + (int(np.prod(self.shape[start_dim:end_dim+1])),)
          + self.shape[end_dim+1:])
        return self.reshape(new_shape)
        
    def squeeze(self: tensor, dim: builtins.int | None = None) -> Self:
        if dim is None: new_shape = tuple(s for s in self.shape if s != 1)
        else:
            if self.shape[dim] != 1: return self 
            new_shape = self.shape[:dim] + self.shape[dim+1:]
        return self.reshape(new_shape)

    def unsqueeze(self: tensor, dim: builtins.int) -> Self:
        dim = dim if dim >= 0 else self.ndim + dim + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(new_shape)
            
    def permute(
        self: tensor, 
        *dims: builtins.int 
             | list[builtins.int] 
             | tuple[builtins.int, ...] 
             | None
    ) -> Self:
        dims = tensor._normalize_dim(dims, self.ndim)
        self_requires_grad = self.requires_grad
    
        if self.device == 'cuda': out_data = cuda.shapes.permute(self, dims)
        else: out_data = cpu.shapes.permute(self.data, dims) 
        out_shape = tuple(self.shape[d] for d in dims)
        
        out = self._new(out_data, out_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims): inv_dims[d] = i
        inv_dims = tuple(inv_dims)
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad.permute(inv_dims)
        
        out._backward = _backward
        return out

    def transpose(
        self: tensor,
        dim1: builtins.int, 
        dim2: builtins.int
    ) -> Self:
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(tuple(dims))

    def swapdims(
        self: tensor, 
        dim1: builtins.int, 
        dim2: builtins.int
    ) -> Self: 
        return self.transpose(dim1, dim2)

    def expand(
        self:   tensor, 
        *shape: builtins.int | tuple[builtins.int, ...] | typing.Size 
    ) -> Self:
        shape = tensor._normalize_shape_input(*shape)

        assert len(shape) == self.ndim, \
            f'expand target shape must have same ndim as input'
        
        orig_shape = self.shape
        assert all(t == s or s == 1 for s, t in zip(orig_shape, shape)), \
            f'expand can only expand size-1 dimensions'
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.shapes.expand(self, shape)
        else: out_data = cpu.shapes.expand(self.data, shape).copy()
        out = self._new(out_data, shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = out.grad
                s = zip(orig_shape, shape)
                for i, (in_dim, out_dim) in enumerate(s):
                    if in_dim == 1 and out_dim != 1:
                        grad = grad.sum(dim=i, keepdim=True)
                self.grad += grad

        out._backward = _backward
        return out

    def broadcast_to(
        self:   tensor,
        *shape: builtins.int | tuple[builtins.int, ...] | typing.Size 
    ) -> Self: 
        return self.expand(*shape)
    
    def unfold(
        self:      tensor, 
        dimension: builtins.int, 
        size:      builtins.int, 
        step:      builtins.int
    ) -> Self:
        
        # TODO: This needs to be made GPU native at some point. Kinda janky
        #       right now because of the multiple im2col/col2im bindings and
        #       the lack of 3d im2col/col2im kernels. Not super performance
        #       critical though, so fine on CPU for the time being.
        
        self_requires_grad = self.requires_grad
        
        dim = dimension if dimension >= 0 else self.ndim + dimension
        L_out = (self.shape[dim] - size) // step + 1
        out_shape = self.shape[:dim] + (L_out,) + self.shape[dim+1:] + (size,)

        arr = self.numpy()
        s = arr.strides
        out_strides = s[:dim] + (s[dim] * step,) + s[dim+1:] + (s[dim],)
        result = np.lib.stride_tricks.as_strided(
            arr, shape=out_shape, strides=out_strides).copy()
        
        out = self._new(result, out_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        def _backward() -> None:
            if self_requires_grad:
                g = out.grad.cpu().numpy() if self.device == 'cuda' \
                    else out.grad.numpy()
                grad_input = np.zeros(self.shape, dtype=np.float32)
                for n in range(L_out):
                    g_idx = tuple(
                        n if i == dim else
                        slice(None)
                        for i in range(len(out_shape) - 1))
                    g_window = g[g_idx]
                    
                    g_window = np.moveaxis(g_window, -1, dim)
                    
                    dst = tuple(
                        slice(n * step, n * step + size) if i == dim else
                        slice(None)
                        for i in range(self.ndim))
                    grad_input[dst] += g_window
                
                grad_tensor = self._new(
                    grad_input, self.shape, typing.float32, self.device)
                self.grad += grad_tensor

        out._backward = _backward
        return out
        
    def flip(self: tensor, dim: builtins.int) -> Self:
        self_requires_grad = self.requires_grad
        dim = dim if dim >= 0 else self.ndim + dim

        if self.device == 'cuda':
              out_data = cuda.shapes.flip(self, dim)
        else: out_data = np.flip(self.data, axis=dim).copy()

        out = self._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad.flip(dim)

        out._backward = _backward
        return out
        
    ### COMBINATION ###
    
    def select(
        self:  tensor, 
        dim:   builtins.int, 
        index: builtins.int
    ) -> Self:
        dim = tensor._normalize_dim(dim, self.ndim)
        idx = [index if i == dim else slice(None) for i in range(self.ndim)]
        return self[tuple(idx)]

    def unstack(self: tensor, dim: builtins.int = 0) -> list[tensor]:
        return [self.select(dim, i) for i in range(self.shape[dim])]
        
    def unbind(self: tensor, dim: builtins.int = 0) -> list[tensor]:
        return self.unstack(dim)
        
    def split(
        self:       tensor, 
        split_size: builtins.int | list[builtins.int], 
        dim:        builtins.int = 0
    ) -> list[tensor]:
        if isinstance(split_size, builtins.int):
            indices = range(0, self.shape[dim], split_size)
            outputs = []
            for start in indices:
                idx = []
                for i in range(self.ndim):
                    if i != dim: idx.append(slice(None))
                    else: idx.append(slice(start, start + split_size))
                outputs.append(self[tuple(idx)])
            return outputs
        else:
            chunks = []
            start = 0
            for size in split_size:
                idx = []
                for i in range(self.ndim):
                    if i != dim: idx.append(slice(None))
                    else: idx.append(slice(start, start + size))
                chunks.append(self[tuple(idx)])
                start += size
            return chunks

    def chunk(
        self: tensor, 
        size: builtins.int,
        dim:  builtins.int = 0
    ) -> list[tensor]:
        assert size >= 1
        chunk_size = builtins.int(np.ceil(self.shape[dim] / size))
        return self.split(chunk_size, dim)
    
    ### INDEXING ###
    
    def gather(
        self:  tensor, 
        dim:   builtins.int | None, 
        index: tensor
    ) -> Self:
        assert index.device == self.device, (
            f'Gather expects input tensor and index tensor to be on same '
            f'device, but found two devices, {self.device} and {index.device}')
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
              out_data = cuda.indexing.gather(self, dim, index)
        else: out_data =  cpu.indexing.gather(self.data, dim, index.data)
        out = self._new(out_data, index.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad = self._new(
                    np.zeros(self.shape, typing.float32), 
                    self.shape, typing.float32, self.device, 
                    requires_grad=False)
                self.grad += grad.scatter_add(dim, index, out.grad)
        
        out._backward = _backward
        return out

    def scatter(
        self:   tensor, 
        dim:    builtins.int, 
        index:  tensor, 
        source: tensor | builtins.int | builtins.float
    ) -> Self:
        if not isinstance(source, tensor):
            source = self._new(
                np.full(index.shape, fill_value=source), 
                index.shape, self.dtype, self.device)
        
        if not self.device == index.device or not self.device == source.device:
            _devices = set([self.device, index.device, source.device])
            raise ValueError(
                f'Scatter expects all tensors to be on same device, but found '
                f'multiple devices: {list(_devices)}')
        assert self.dtype == source.dtype, \
            f'Input and source must have the same dtype.'
        
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)
            
        self_requires_grad = self.requires_grad
        source_requires_grad = source.requires_grad
        
        if self.device == 'cuda':
              out_data = cuda.indexing.scatter(self, dim, index, source)
        else: out_data =  cpu.indexing.scatter(
            self.data, dim, index.data, source.data)
        out = self._new(out_data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self, source))

        def _backward() -> None:
            if source_requires_grad:
                source.grad += out.grad.gather(dim, index)
            if self_requires_grad:
                mask = self._new(
                    np.ones(self.shape, typing.float32),
                    self.shape, typing.float32, self.device,
                    requires_grad=False)
                mask = mask.scatter(dim, index, 0.0)
                self.grad += out.grad * mask             
            
        out._backward = _backward
        return out

    def index_select(
        self:  tensor, 
        dim:   builtins.int, 
        index: tensor
    ) -> Self:
        dim = tensor._normalize_dim(dim, self.ndim)
        idx_shape = [1] * self.ndim
        idx_shape[dim] = len(index)
        index = index.reshape(tuple(idx_shape))
        
        gather_shape = list(self.shape)
        gather_shape[dim] = index.shape[dim]
        index = index.expand(tuple(gather_shape))
        
        return self.gather(dim, index) 


