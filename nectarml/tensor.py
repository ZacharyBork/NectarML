from __future__ import annotations

import warnings
import builtins
from typing import Any, Literal
from collections.abc import Callable

import numpy as np

from nectarml import typing, cpu, cuda, autograd
from nectarml.cuda.memory import CudaBuffer
from nectarml.amp.precision import amp_float16, amp_float32

class tensor:
    _class_type_nectar_tensor = True
    
    def __init__(
        self:          tensor,
        data:          np.ndarray | CudaBuffer,
        shape:         typing.Size | tuple[builtins.int, ...],
        dtype:         typing.DTypeLike,
        device:        Literal['cpu', 'cuda'],
        requires_grad: bool,
        _children:     tuple[tensor, ...]
    ) -> None:
        '''Initializes a new tensor object.
        
        Args:
            data : Either an np.ndarray object of tensor data, or a CudaBuffer
                pointing to the tensor's data in CUDA memory.
            shape : A tuple[int, ...] or Size object defining the Tensor's
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
        self.device = device
        self._dtype = dtype
        self.shape  = shape
        self.requires_grad = requires_grad
        
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
        cls,
        data: builtins.int | np.ndarray,
        shape: typing.Size | tuple[builtins.int, ...],
        dtype: typing.DTypeLike,
        device: Literal['cpu', 'cuda'],
        requires_grad: bool = False,
        _children: tuple = ()
    ) -> tensor:
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
            assert isinstance(data, np.ndarray), \
                f'_new() for CPU tensors requires data to be of type ' \
                f'"np.ndarray", but recieved type {type(data)}'
            out._buffer = None
            out.data = data

        return out
    
    @classmethod
    def _from_data(
        cls: type[tensor],
        data: np.ndarray, 
        shape: typing.Size | tuple[builtins.int, ...], 
        dtype: typing.DTypeLike, 
        requires_grad: bool = False,
        _children: tuple = ()
    ) -> tensor:
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
        cls,
        buffer: CudaBuffer,
        shape: typing.Size,
        dtype: typing.DTypeLike,
        requires_grad: bool = False,
        _children: tuple = ()
    ) -> tensor:
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
    
    ### PROPERTIES ###
      
    @property
    def _data_ptr(self: tensor) -> builtins.int | None:
        '''Property access for CUDA pointer to tensor's data.
        
        Returns:
            int | None : If the given tensor owns a CudaBuffer (i.e. if its
                device is "cuda"), returns the uintptr to the tensor's data in
                CUDA memory. Otherwise returns NoneType.
        '''
        if self._buffer is not None: return self._buffer.ptr
        return None
      
    @property
    def dtype(self: tensor) -> typing.DTypeLike:
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
    def size(self: tensor) -> builtins.int:
        '''Propert access for tensor size (i.e. numel).
        
        Returns:
            int : The number of elements in the given number shape. Equivelent
                to math.prod(tensor.shape).
        '''
        return self.shape.numel()
           
    @property
    def is_contiguous(self) -> bool:
        if self.device == 'cuda': return True
        else: return self.data.flags['C_CONTIGUOUS']
            
    ### DATA UTILS ###
    
    def numpy(self: tensor) -> np.ndarray:
        '''Returns the tensor's data as a numpy.ndarray.
        
        Returns:
            np.ndarray : The tensor's data as a numpy.ndarray.
        '''
        if self.device == 'cuda': 
            return cuda.to_cpu(self)
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
    
    def contiguous(self) -> tensor:
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
    
    def clone(self: tensor) -> tensor:
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
    
    def copy_(self: tensor, other: tensor) -> tensor:
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
            
    ### DEVICE / DTYPE ###
        
    def to(
        self: tensor,
        device: Literal['cpu', 'cuda'] | None = None,
        dtype: typing.DTypeLike | None = None
    ) -> tensor: 
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
                data = cuda.to_cuda(src)
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
            orig_device = self.device
            orig_dtype  = self.dtype
            
            def _backward() -> None:
                if self.requires_grad:
                    self.grad += new.grad.to(orig_device, orig_dtype)
            
            new._prev     = {self}
            new._backward = _backward

        return new
            
    def cuda(self: tensor) -> tensor: 
        '''Convenience function to cast given tensor to CUDA device.
        
        When called on a tensor who's device is already "cuda", this method 
        will return a reference to the original tensor object. If you would 
        like to make a duplicate of a given tensor, please see tensor.clone().
        
        Returns:
            tensor : The resulting CUDA tensor from the cast operation.
        '''
        return self.to(device='cuda')

    def cpu(self: tensor) -> tensor: 
        '''Convenience function to cast given tensor to CPU device.
        
        When called on a tensor who's device is already "cpu", this method 
        will return a reference to the original tensor object. If you would 
        like to make a duplicate of a given tensor, please see tensor.clone().
        
        Returns:
            tensor : The resulting CPU tensor from the cast operation.
        '''
        return self.to(device='cpu')
    
    ### GETTERS / SETTERS ###

    def __getitem__(
        self: tensor, 
        idx: (
            tensor 
          | tuple[tensor, tensor]
          | builtins.int
          | slice
          | tuple[slice])
    ) -> tensor:
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
            
            K = idx0.numel()
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
        self: tensor, 
        idx: builtins.int | slice | tuple[slice],
        value: Any
    ) -> None:
        warnings.warn(
            'tensor.__setitem__() modifies the given tensor in-place. This '
            'operation does not support autograd. Calling this method on a '
            'tensor in the autograd graph will corrupt the gradients of any '
            'tensor which depends on it.')
        
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
    
    def __len__(self) -> builtins.int: 
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
    
    def reshape(self: tensor, shape: tuple[builtins.int, ...]) -> tensor:
        self_requires_grad = self.requires_grad
        orig_shape = self.shape
        
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
    
    def view(self: tensor, shape: tuple[builtins.int, ...]) -> tensor:
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
    ) -> tensor:
        end_dim = end_dim if end_dim >= 0 else self.ndim + end_dim
        new_shape = (
            self.shape[:start_dim]
          + (int(np.prod(self.shape[start_dim:end_dim+1])),)
          + self.shape[end_dim+1:])
        return self.reshape(new_shape)
        
    def squeeze(self: tensor, dim: builtins.int | None = None) -> tensor:
        if dim is None: new_shape = tuple(s for s in self.shape if s != 1)
        else:
            if self.shape[dim] != 1: return self 
            new_shape = self.shape[:dim] + self.shape[dim+1:]
        return self.reshape(new_shape)

    def unsqueeze(self: tensor, dim: builtins.int) -> tensor:
        dim = dim if dim >= 0 else self.ndim + dim + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(new_shape)
            
    def permute(self: tensor, dims: tuple[builtins.int, ...] | None) -> tensor:
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
    ) -> tensor:
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(tuple(dims))

    def swapdims(
        self: tensor, 
        dim1: builtins.int, 
        dim2: builtins.int
    ) -> tensor: 
        return self.transpose(dim1, dim2)

    def expand(self: tensor, shape: tuple[builtins.int, ...]) -> tensor:
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
                        grad = grad.sum(dim=i, keepdims=True)
                self.grad += grad

        out._backward = _backward
        return out

    def broadcast_to(self: tensor, shape: tuple[builtins.int, ...]) -> tensor:
        return self.expand(shape)
    
    def unfold(
        self:      tensor, 
        dimension: builtins.int, 
        size:      builtins.int, 
        step:      builtins.int
    ) -> tensor:
        
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
                    grad_input, self.shape, self.dtype, self.device)
                self.grad += grad_tensor

        out._backward = _backward
        return out
        
    def flip(self: tensor, dim: builtins.int) -> tensor:
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
    
    def select(self: tensor, dim: builtins.int, index: builtins.int) -> tensor:
        idx = [index if i == dim else slice(None) for i in range(self.ndim)]
        return self[tuple(idx)]

    def unstack(self: tensor, dim: builtins.int = 0) -> list[tensor]:
        return [self.select(dim, i) for i in range(self.shape[dim])]
        
    def unbind(self: tensor, dim: builtins.int = 0) -> list[tensor]:
        return self.unstack(dim)
        
    def split(
        self: tensor, 
        split_size: builtins.int | list[builtins.int], 
        dim: builtins.int = 0
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
    ) -> tensor:
        assert index.device == self.device, (
            f'Gather expects input tensor and index tensor to be on same '
            f'device, but found two devices, {self.device} and {index.device}')
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            out_data = cuda.indexing.gather(self, dim, index)
        else: out_data = cpu.indexing.gather(self.data, dim, index.data)
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
    ) -> tensor:
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
            out_data = cuda.indexing.scatter(self, index, dim, source)
        else: out_data = cpu.indexing.scatter(
            self.data, index.data, dim, source.data)
        out = self._new(out_data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))

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
    ) -> builtins.int:
        idx_shape = [1] * self.ndim
        idx_shape[dim] = len(index)
        index = index.reshape(tuple(idx_shape))
        
        gather_shape = list(self.shape)
        gather_shape[dim] = index.shape[dim]
        index = index.expand(tuple(gather_shape))
        
        return self.gather(dim, index)
    
    
    
    
###############################################################################
###############################################################################
###############################################################################
    
class BoolTensor(tensor):    
    def __init__(
        self:   BoolTensor,
        data:   typing.ArrayLike | None = None,
        shape:  typing.Size | tuple[builtins.int, ...] | None = None,
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        if not isinstance(data, np.ndarray): data = np.array(data)
        shape = shape if isinstance(shape, typing.Size) else \
                typing.Size(shape or data.shape)
        data  = self._build_data(data, shape, device)
        super().__init__(data, shape, typing.bool_, device)

    def _build_data(
        self:   BoolTensor, 
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
    
class Tensor(tensor):    
    def __init__(
        self: Tensor,
        data: typing.ArrayLike,
        shape: typing.Size | tuple[builtins.int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        '''Initializes a new Tensor object.
        
        If data is an ArrayLike object and the Tensor's device is "cpu", this
        method will fill the Tensor's data with the data from the ArrayLike
        object. In this case, if shape is provided, it will be used to set the
        Tensor's shape. If shape is not provided, the Tensor's shape will
        instead be taken from the shape of the input data.
        
        If the Tensor's device is "cuda" and data is a unintptr to tensor data
        in CUDA memory, this method will create a CudaBuffer object from the
        given pointer. In this case, shape is required, and will be used
        directly to set the Tensor's shape.
        
        If the Tensor's device is "cuda" and data is an ArrayLike object, the
        given data will be passed to CUDA and a CudaBuffer object will be
        created for the resulting pointer. In this case, shape is not required,
        and if not provided, will be assumed from the shape of the given data.
        Shape may be provided, however, and if it is, will override the shape
        of the input data when setting the Tensor's shape.
        
        Args:
            data : Either an ArrayLike object of tensor data, or a uintptr to
                a tensor in CUDA memory.
            shape : Optional if initializing with ArrayLike data. Used to
                define the shape of the Tensor.
            dtype : A DtypeLike defining the data type for the new Tensor.
            device : The device for the new Tensor, "cpu" or "cuda".
            requires_grad : A boolean defining whether the new Tensor should
                require grad or not. If True, a grad Tensor will be created
                and assigned to the new Tensor, and the new Tensor will be
                included in the computational graph to participate in gradient
                backpropagation. If False, the Tensor will not be included in
                the computational graph, and will not contribute to the 
                network's gradients.
            _children : The _prev Tensors for the newly created Tensor. If the
                new Tensor is included in the computational graph, the 
                gradients from the new Tensor will flow back to the _children
                Tensors during backpropagation. Used for autograd operations,
                generally not set manually.
        '''
        if not isinstance(data, np.ndarray): data = np.array(data)
        data = data.astype(dtype)
        
        shape = shape if isinstance(shape, typing.Size) else \
                typing.Size(shape or data.shape)
        data  = self._build_data(data, shape, dtype, device)
        super().__init__(data, shape, dtype, device, requires_grad, _children)

    ### INIT ###
        
    def _build_data(
        self:   Tensor, 
        data:   np.ndarray,
        shape:  typing.Size,
        dtype:  typing.DTypeLike,
        device: Literal['cpu', 'cuda'] 
    ) -> np.ndarray | CudaBuffer:
        '''Initializes a Tensor from given data and optional shape.
        
        Args:
            data : Either an ArrayLike object of tensor data, or a uintptr to
                a tensor in CUDA memory.
            shape : Optional if initializing with ArrayLike data. Used to
                define the shape of the Tensor.
                
        Raises:
            ValueError : If data is uintptr to CUDA tensor data and shape is
                not provided.
            ValueError : If Tensor's device type is not valid (i.e. not "cpu"
                or "cuda").
        '''
        match device:
            case 'cpu': ref = data
            case 'cuda': 
                ref = CudaBuffer(
                    cuda.data_to_cuda(data, shape.numel(), dtype), dtype)
            case _: raise ValueError(f'Invalid device type: {device}')
        return ref
    
    ### PROPERTIES ###
    
    @property
    def requires_grad(self: tensor) -> bool:
        '''Property access for tensor's requires_grad value.
    
        Returns:
            bool : True if given tensor requires grad, otherwise False.
        '''
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self: tensor, value: bool) -> None:
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
        
    ### UTILS ###

    def detach(self: Tensor) -> Tensor:
        '''Returns a copy of the tensor detached from the computation graph.
        
        NOTE: The newly created tensor will share the same same underlying
        storage. As such, modifying the resulting detached tensor in-place will 
        also modify the original tensor.
        
        Returns:
            tensor : A detached copy of the tensor this method is called on.
        '''
        if self.device == 'cuda':
            return self._from_buffer(
                self._buffer, self.shape, self.dtype, self.requires_grad)
        else: 
            return self._from_data(
                self.data.view(), self.shape, self.dtype, self.requires_grad)
    
    def detach_(self: Tensor) -> None:
        '''In-place detach. Detaches given tensor from the computation graph. 
        
        WARNING: This will corrupt the gradients of any tensors which depend on 
        the tensor this is called from!
        
        Detaches the tensor this method is called on from the computation graph
        by disabling requires_grad and clearing all autograd data.
        '''
        self.requires_grad = False
        self._backward = None
        self._prev.clear()
    
    def requires_grad_(self: Tensor, value: bool) -> Tensor:
        '''In-place setter for tensor.requires_grad.
        
        If value=True and the given tensor does not already have a grad tensor,
        this method will allocate a new grad tensor. If value=False and the
        tensor does have a grad tensor, the grad tensor will be deallocated and
        the memory will be freed (or buffer decremented, in the case of CUDA
        tensors).
        
        NOTE: Calling this function with value=True inside of a no_grad context
        will bypass the context and set requires_grad=True.
        
        Args:
            value : The new value for requires_grad.
            
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        self.requires_grad = value
        if value and self.grad is None: self._allocate_grad()
        elif not value: self._deallocate_grad()
        return self
    
    def fill_(
        self: Tensor, 
        fill_value: builtins.float | builtins.int
    ) -> Tensor:
        '''In-place fill method. Fills given tensor data with fill_value.
        
        Args:
            fill_value : The (float|int) value to fill the tensor's data with.
            
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        if self.device == 'cuda':
            new_ptr = cuda.memory.alloc_cuda_full(
                self.size, self.dtype, fill_value)
            old_buffer = self._buffer
            self._buffer = CudaBuffer(new_ptr, self.dtype)
            old_buffer.decrement()
        else: self.data.fill(fill_value)
        return self
    
    def zero_(self: Tensor) -> Tensor:
        '''In-place zero-fill method. Fills given tensor's data with zeros.
        
        Returns:
            tensor : A reference to the tensor that this method was called on.
                Useful for chaining ops.
        '''
        return self.fill_(0.0)

    def _broadcast_shape(
        self,
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
    
    def _broadcast_grad(
        self,
        grad: Tensor,
        original_shape: tuple[builtins.int, ...] | typing.Size, 
    ) -> Tensor:
        if grad.shape == original_shape: return grad

        ndim_diff = grad.ndim - len(original_shape)
        axes = list(range(ndim_diff))
        shapes = zip(original_shape, grad.shape[ndim_diff:])
        for i, (s, g) in enumerate(shapes):
            if s != g: axes.append(i + ndim_diff)
        
        if axes: grad = grad.sum(dim=tuple(axes), keepdim=False)
        if grad.shape != original_shape: grad = grad.reshape(original_shape)
        return grad

    def _build_output_tensor(
        self: Tensor, 
        data: np.ndarray | builtins.int, 
        children: tuple[Tensor, ...]
    ) -> Tensor:
        '''Helper method to build output tensor's from tensor ops.
        
        Args:
            data : Either ArrayLike data for new tensor, or uintptr to new
                tensor's data in CUDA memory.
            children : The child tensors to assign to the newly created tensor
                for backpropagation.
        '''
        _requires_grad = False
        for child in children:
            if child.requires_grad: _requires_grad = True
        return self._new(data, self.shape, self.dtype, self.device, 
            _requires_grad, _children=children)
        
    ### GRADIENTS ###
    
    def _deallocate_grad(self: Tensor) -> None:
        '''Deallocated grad by setting tensor.grad to None.'''
        self.grad = None

    def _allocate_grad(self: Tensor, fill_value: builtins.float = 0.0) -> None:
        '''Allocates gradient tensor for given tensor.
        
        Args:
            fill_value : The value to fill the new grad tensor with.
        '''
        self._deallocate_grad()
        self.grad = self._new(
            np.full(self.shape, fill_value, typing.float32), 
            self.shape, typing.float32, self.device, requires_grad=False) 
        
    def backward(self: Tensor) -> None:
        '''Gradient backpropagation method.
        
        Builds a topo graph from all children of the tensor it was called on
        recursively. Then allocates a new gradient tensor for the given tensor,
        fills the gradient tensor with ones, and then walks the graph, calling
        each subsequent child's _backward() method.
        '''
        assert self.ndim == 0 or self.size == 1, \
            'backward() can only be called on scalar tensors.'
        
        visited: set[builtins.int] = set()
        graph: list[tensor] = []
        
        def build_graph(node: tensor):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_graph(child)
                graph.append(node)
        
        build_graph(self)
        graph.reverse()
        
        for node in graph:
            if node.requires_grad and node is not self:
                node._allocate_grad(fill_value=0.0)
                    
        self._allocate_grad(fill_value=1.0)
        for node in graph: node._backward()
        
        for node in graph:
            if node._prev:
                node._prev.clear()
                node._backward = lambda : None
    
    def zero_grad(self: Tensor) -> None:
        '''Zeros values in the grad tensor of the tensor it is called on.'''
        if self.requires_grad and self.grad is not None:
            self.grad.zero_()
    
    ### COMPARISON ###
    
    def __eq__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
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
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda': 
            data = cuda.math.equal(self, other, out_shape)
        else: data = cpu.math.equal(self, other)
        
        return BoolTensor._new(data, out_shape, typing.bool_, self.device)
        
    def __lt__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise less than operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is less 
                than to the corresponding value in the "other" tensor. Note: 
                This return tensor can be turned into a binary mask like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x < y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda':
            data = cuda.math.less_than(self, other, out_shape)
        else: data = cpu.math.less_than(self, other)
        return BoolTensor._new(data, out_shape, typing.bool_, self.device)
    
    def __le__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise less than or equal operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is less 
                than or equal to the corresponding value in the "other" tensor. 
                Note: This return tensor can be turned into a binary mask 
                like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x <= y).to(x.device, x.dtype)
        '''
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape

        if self.device == 'cuda':
            data = cuda.math.less_than_or_equal(self, other, out_shape)
        else: data = cpu.math.less_than_or_equal(self, other)
        return BoolTensor._new(data, out_shape, typing.bool_, self.device)
    
    def __gt__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise greater than operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is greater 
                than to the corresponding value in the "other" tensor. Note: 
                This return tensor can be turned into a binary mask like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x > y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda':
            data = cuda.math.greater_than(self, other, out_shape)
        else: data = cpu.math.greater_than(self, other)
        return BoolTensor._new(data, out_shape, typing.bool_, self.device)
    
    def __ge__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> BoolTensor:
        '''tensor elementwise greater than or equal operator.
        
        Args:
            other : The tensor, int, or float value to compare the given tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new tensor with the same shape, device,
                and Dtype as the tensor this operator is called on.
                
        Returns:
            tensor : A boolean tensor denoting whether each element is greater 
                than or equal to the corresponding value in the "other" tensor. 
                Note: This return tensor can be turned into a binary mask 
                like so:
                
                    x = tensor(data, dtype=typing.float32)
                    y = tensor(data, dtype=typing.float32)
                    mask = (x >= y).to(x.device, x.dtype)
        '''        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
        else: out_shape = self.shape
        
        if self.device == 'cuda':
            data = cuda.math.greater_than_or_equal(self, other, out_shape)
        else: data = cpu.math.greater_than_or_equal(self, other)
        return BoolTensor._new(data, out_shape, typing.bool_, self.device)
    
    def __hash__(self: tensor) -> builtins.int: 
        '''Hash override. Returns memory address of tensor.
        
        Returns:
            int : Address in system memory of the tensor.
        '''
        return id(self)
    
    ### ROUNDING ###
    
    def floor(self) -> Tensor:
        '''Takes the floor of the given tensor's data, returns as new tensor.
        
        Returns:
            tensor : The resulting tensor from the floor operation.
        '''
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.floor(self)
        else: out_data = np.floor(self.data).astype(self.dtype)
        out = self._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
                
        out._backward = _backward
        return out
    
    def ceil(self) -> Tensor:
        '''Takes the ceil of the given tensor's data, returns as new tensor.
        
        Returns:
            tensor : The resulting tensor from the ceil operation.
        '''
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.ceil(self)
        else: out_data = np.ceil(self.data).astype(self.dtype)
        out = self._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
                
        out._backward = _backward
        return out
    
    def round(self, precision: builtins.int = 0) -> Tensor:
        '''Rounds the data of a given tensor and returns as new tensor.
        
        NOTE: Currently, the precision argument only works on CPU tensors. CUDA
        tensors will be rounded to the nearest whole integer (though they will
        keep the DType of the original tensor).
        
        Args:
            precision : The number of decimal places to round the tensor's data
                to (only works on CPU tensors currently).
        
        Returns:
            tensor : The resulting tensor from the rounding operation.
        '''
        
        # CUDA ROUND NEEDS PRECISION!
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.round(self)
        else: out_data = np.round(self.data, decimals=precision)
        out = self._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
                
        out._backward = _backward
        return out
    
    ### MATH DUNDERS ###
    
    def __iadd__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Adds other (tensor|int|float) to given tensor's data in-place.
        
        Args:
            other : The other tensor or scalar value to add to the tensor.
            
        Returns:
            tensor : A reference to the tensor being added to.
        '''
        if isinstance(other, Tensor):
            assert other.shape == self.shape, \
                f'grad shape mismatch: self.shape={self.shape} ' \
                f'other.shape={other.shape}'

        if self.device == 'cuda': 
            new_ptr = cuda.math.add(self, other, self.shape)
            self._buffer.decrement()
            self._buffer = CudaBuffer(new_ptr, self.dtype)
        else: self.data = cpu.math.add(self, other)
        return self
    
    def __add__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Adds (tensor|int|float) to tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
            
        if self.device == 'cuda': 
            out_data = cuda.math.add(self, other, out_shape)
        else: out_data = cpu.math.add(self, other)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += self._broadcast_grad(out.grad, self.shape)
            if other_requires_grad:
                other.grad += self._broadcast_grad(out.grad, other.shape)
                
        out._backward = _backward
        return out
    
    def __radd__(
        self: Tensor, 
        other: tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Adds (tensor|int|float) to tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''
        return self + other

    def __isub__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Subtracts other (tensor|scalar) from given tensor's data in-place.
        
        Args:
            other : The other tensor or scalar value to subtract from the 
                tensor.
            
        Returns:
            tensor : A reference to the tensor being subtracted from.
        '''
        if isinstance(other, Tensor):
            assert other.shape == self.shape, \
                f'grad shape mismatch: self.shape={self.shape} ' \
                f'other.shape={other.shape}'
        
        if self.device == 'cuda': 
            new_ptr = cuda.math.subtract(self, other, self.shape)
            self._buffer.decrement()
            self._buffer = CudaBuffer(new_ptr, self.dtype)    
        else: self.data = cpu.math.subtract(self, other)
        return self

    def __sub__(self, other: Tensor | builtins.int | builtins.float) -> Tensor:
        '''Subtracts (tensor|scalar) from tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''        
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.subtract(self, other, out_shape)   
        else: out_data = cpu.math.subtract(self, other)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self._broadcast_grad(out.grad, self.shape)
            if other_requires_grad: 
                other.grad += -self._broadcast_grad(out.grad, other.shape)

        out._backward = _backward
        return out
    
    def __rsub__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Subtracts (tensor|scalar) from tensor's data and returns new tensor.
        
        Args:
            other : The tensor or scalar value to add to the given tensor.
            
        Returns:
            tensor : Resulting tensor from addition operation.
        '''
        return (-self) + other
    
    def __neg__(self) -> Tensor:
        '''Negates the data of a given tensor.
            
        Returns:
            tensor : A new tensor with the data from the negation operation.
        '''
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.negate(self)
        else: out_data = cpu.math.negate(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += -out.grad

        out._backward = _backward
        return out

    def __imul__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Multiplies other (tensor|scalar) with given tensor's data in-place.
        
        Args:
            other : The other tensor or scalar value to multiply the tensor by.
            
        Returns:
            tensor : A reference to the tensor being multiplied.
        '''
        if isinstance(other, Tensor):
            assert other.shape == self.shape, \
                f'grad shape mismatch: self.shape={self.shape} ' \
                f'other.shape={other.shape}'
        
        if self.device == 'cuda': 
            new_ptr = cuda.math.multiply(self, other, self.shape)
            self._buffer.decrement()
            self._buffer = CudaBuffer(new_ptr, self.dtype)
        else: self.data = cpu.math.multiply(self, other)
        return self

    def __mul__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Multiplies tensor by (tensor|scalar) and returns new tensor.
        
        Args:
            other : The tensor or scalar value to multiply the given tensor by.
            
        Returns:
            tensor : Resulting tensor from multiplication operation.
        '''
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.multiply(self, other, out_shape)
        else: out_data = cpu.math.multiply(self, other)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad: 
                grad = other * out.grad
                self.grad += self._broadcast_grad(grad, self.shape)
            if other_requires_grad: 
                grad = self * out.grad
                other.grad += self._broadcast_grad(grad, other.shape)
        
        out._backward = _backward
        return out
    
    def __rmul__(
        self: Tensor, 
        other: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        '''Multiplies tensor by (tensor|scalar) and returns new tensor.
        
        Args:
            other : The tensor or scalar value to multiply the given tensor by.
            
        Returns:
            tensor : Resulting tensor from multiplication operation.
        '''
        return self * other
    
    @amp_float16
    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        '''Performs a matrix multiplication between the data of two tensors.
        
        Args:
            other : The other tensor for the matrix multiplication.
            
        Returns:
            tensor : Resulting tensor from matrix multiplication operation.
        '''
        self._validate_other(other)
        if self.ndim == 1 or other.ndim == 1:
            raise NotImplementedError('matmul not supported for 1D tensors.')
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.matmul.matmul(self, other)
        else: out_data = cpu.math.matmul(self, other)
        out = self._build_output_tensor(out_data, (self, other))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad @ other.transpose(-2, -1)
            if other_requires_grad:
                other.grad += self.transpose(-2, -1) @ out.grad
        
        out._backward = _backward
        return out
    
    def __rmatmul__(self: Tensor, other: Tensor) -> Tensor: 
        '''Performs a matrix multiplication between the data of two tensors.
        
        Args:
            other : The other tensor for the matrix multiplication.
            
        Returns:
            tensor : Resulting tensor from matrix multiplication operation.
        '''
        return other @ self
    
    @amp_float32
    def __pow__(
        self: Tensor, 
        exponent: builtins.float | builtins.int
    ) -> Tensor:
        '''Raises a tensor by the given exponent and returns as new tensor.
        
        Args:
            exponent : The exponent to raise the tensor's data by.
            
        Returns:
            tensor : Resulting tensor from power operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.pow(self, exponent)
        else: out_data = cpu.math.pow(self, exponent)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += exponent * (self**(exponent-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __rpow__(
        self: Tensor, 
        exponent: builtins.float | builtins.int
    ) -> Tensor: 
        raise NotImplementedError
    
    def __truediv__(
        self: Tensor, 
        other: Tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Divides a tensor by a (tensor/scalar) and returns as new tensor.
        
        Args:
            other : The (tensor/scalar) to act as the divisor.
            
        Returns:
            tensor : Resulting tensor from division operation.
        ''' 
        return self * other ** -1
    
    def __rtruediv__(
        self: Tensor, 
        other: Tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Divides a tensor by a (tensor/scalar) and returns as new tensor.
        
        Args:
            other : The (tensor/scalar) to act as the divisor.
            
        Returns:
            tensor : Resulting tensor from division operation.
        ''' 
        return (self ** -1) * other
    
    def __abs__(self: Tensor) -> Tensor:
        '''Takes the absolute of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from absolute operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.abs(self)
        else: out_data = cpu.math.abs(self)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self.sign() * out.grad
        
        out._backward = _backward
        return out
    
    ### CLAMP ###
    
    def minimum(
        self: Tensor, 
        other: Tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Return the minimum of a tensor and a (tensor/scalar) as new tensor.
        
        NOTE: It "other" is a tensor, this method will perform and elementwise
        mimimum operation, comparing each element of the given tensor against 
        the corresponding element from the "other" tensor and returning the
        smaller value.
        
        Args:
            other : The (tensor/scalar) to compare against.
        
        Returns:
            tensor : Resulting tensor from the minimum operation.
        '''         
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.minimum(self, other, out_shape)
        else: out_data = cpu.math.minimum(self, other)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = (self < other).to(self.device, self.dtype) * out.grad
                self.grad += self._broadcast_grad(grad, self.shape) 
            if other_requires_grad:
                grad = (other < self).to(other.device, other.dtype) * out.grad
                other.grad += self._broadcast_grad(grad, other.shape) 
        
        out._backward = _backward
        return out
    
    def maximum(
        self: Tensor, 
        other: tensor | builtins.float | builtins.int
    ) -> Tensor:
        '''Return the maximum of a tensor and a (tensor/scalar) as new tensor.
        
        NOTE: It "other" is a tensor, this method will perform and elementwise
        maximum operation, comparing each element of the given tensor against 
        the corresponding element from the "other" tensor and returning the
        larger value.
        
        Args:
            other : The (tensor/scalar) to compare against.
        
        Returns:
            tensor : Resulting tensor from the maximum operation.
        '''         
        self_requires_grad = self.requires_grad
        
        if isinstance(other, Tensor):
            out_shape = self._broadcast_shape(self.shape, other.shape)
            children = (self, other)
            other_requires_grad = other.requires_grad
        else:
            out_shape = self.shape
            children = (self,)
            other_requires_grad = False
        
        if self.device == 'cuda': 
            out_data = cuda.math.maximum(self, other, out_shape)
        else: out_data = cpu.math.maximum(self, other)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=children)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = (self >= other).to(self.device, self.dtype) * out.grad
                self.grad += self._broadcast_grad(grad, self.shape) 
            if other_requires_grad:
                grad = (other <= self).to(other.device, other.dtype) * out.grad
                other.grad += self._broadcast_grad(grad, other.shape)
        
        out._backward = _backward
        return out
    
    def clamp(
        self: Tensor, 
        min_value: builtins.float | None = None, 
        max_value: builtins.float | None = None
    ) -> Tensor:
        '''Clamps tensor's values between min and max, returns as new tensor.
        
        Args:
            min_value : The minimum allowable value for the clamping operation,
                or None. If this is None, the minimum representable value for
                the tensor's datatype will be used instead.
            max_value : The minimum allowable value for the clamping operation,
                or None. If this is None, the maximum representable value for
                the tensor's datatype will be used instead.
        
        Returns:
            tensor : Resulting tensor from the clamp operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': 
            out_data = cuda.math.clamp(self, min_value, max_value)
        else: out_data = cpu.math.clamp(self, min_value, max_value)
        out = self._build_output_tensor(out_data, (self,))
                    
        def _backward() -> None:
            if self_requires_grad:
                mask = (self >= min_value).to(self.device, self.dtype) \
                     * (self <= max_value).to(self.device, self.dtype)
                self.grad += mask * out.grad
        
        out._backward = _backward
        return out
        
    ### ABS ###

    def abs(self: Tensor) -> Tensor: 
        '''Takes the absolute of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the absolute operation.
        ''' 
        return self.__abs__()
        
    ### EXP ###
            
    @amp_float32
    def exp(self: Tensor) -> Tensor:
        '''Takes the exponent of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the exponent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.exp(self)
        else: out_data = cpu.math.exp(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out * out.grad

        out._backward = _backward
        return out
      
    ### LOG ###
            
    @amp_float32
    def log(self: Tensor) -> Tensor:
        '''Takes the logarithm of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the log operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log(self)
        else: out_data = cpu.math.log(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / self) * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def log2(self: Tensor) -> Tensor:
        '''Takes the log^2 of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the log2 operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log2(self)
        else: out_data = cpu.math.log2(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(2))

        out._backward = _backward
        return out
    
    @amp_float32
    def log10(self: Tensor) -> Tensor:
        '''Takes the log^10 of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the log10 operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log10(self)
        else: out_data = cpu.math.log10(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(10))

        out._backward = _backward
        return out
          
    ### SQRT ###
            
    def sqrt(self: Tensor) -> Tensor:
        '''Takes the square root of a tensor's data and returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the square root operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.sqrt(self)
        else: out_data = cpu.math.sqrt(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / (2 * out)) * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def rsqrt(self: Tensor) -> Tensor:
        '''Takes the reciprocal sqrt of a tensor's data, returns as new tensor.
        
        Returns:
            tensor : Resulting tensor from the reciprocal square root 
                operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.rsqrt(self)
        else: out_data = cpu.math.rsqrt(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += -0.5 * out**3 * out.grad

        out._backward = _backward
        return out
    
    ### SIN / COS ###
            
    def sin(self: Tensor) -> Tensor:
        '''Returns the sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the sine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.sin(self)
        else: out_data = cpu.math.sin(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            # NOTE: This builds unneeded graph nodes. Eventually, this should
            # be replaced with something akin to torch.no_grad()
            if self_requires_grad: self.grad += self.cos() * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def asin(self: Tensor) -> Tensor:
        '''Returns the arc sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arcsine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.asin(self)
        else: out_data = cpu.math.asin(self)
        
        def _backward() -> None:
            if self_requires_grad:
                denom = (1 - self ** 2).sqrt().clamp(min_value=1e-7)
                self.grad += out.grad / denom
        
        out = self._build_output_tensor(out_data, (self,))
        out._backward = _backward
        return out
    
    @amp_float32
    def sinh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic sine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.sinh(self)
        else: out_data = cpu.math.sinh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += self.cosh() * out.grad
        
        out._backward = _backward
        return out
    
    def asinh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic arc sine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic arcsine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.asinh(self)
        else: out_data = cpu.math.asinh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad / (self**2 + 1).sqrt()
        
        out._backward = _backward
        return out
        
    def cos(self: Tensor) -> Tensor: 
        '''Returns the cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.cos(self)
        else: out_data = cpu.math.cos(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            # NOTE: This builds unneeded graph nodes. Eventually, this should
            # be replaced with something akin to torch.no_grad()
            if self_requires_grad: self.grad += -self.sin() * out.grad

        out._backward = _backward
        return out
    
    @amp_float32
    def acos(self: Tensor) -> Tensor:
        '''Returns the arc cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arc cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.acos(self)
        else: out_data = cpu.math.acos(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = (1 - self**2).sqrt().clamp(min_value=1e-7)
                self.grad += -out.grad / grad
        
        out._backward = _backward
        return out
    
    @amp_float32
    def cosh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.cosh(self)
        else: out_data = cpu.math.cosh(self)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += self.sinh() * out.grad
        
        out._backward = _backward
        return out
    
    def acosh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic arc cosine of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic arc cosine operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.acosh(self)
        else: out_data = cpu.math.acosh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = (self**2 - 1).sqrt().clamp(min_value=1e-7)
                self.grad += out.grad / grad
        
        out._backward = _backward
        return out
    
    ### TAN / ATAN ###
    
    @amp_float32
    def tan(self: Tensor) -> Tensor:
        '''Returns the tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the tangent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.tan(self)
        else: out_data = cpu.math.tan(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += (1 + out**2) * out.grad
        
        out._backward = _backward
        return out
        
    def tanh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic tangent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.tanh(self)
        else: out_data = cpu.math.tanh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += (1 - out**2) * out.grad

        out._backward = _backward
        return out
    
    def atan(self: Tensor) -> Tensor:
        '''Returns the arc tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arc tangent operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atan(self)
        else: out_data = cpu.math.atan(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += out.grad / (1 + self**2)

        out._backward = _backward
        return out
    
    def atanh(self: Tensor) -> Tensor:
        '''Returns the hyperbolic arc tangent of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the hyperbolic arc tangent
                operation.
        ''' 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atanh(self)
        else: out_data = cpu.math.atanh(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += out.grad / (1 - self**2).clamp(min_value=1e-7)

        out._backward = _backward
        return out
    
    def atan2(self: Tensor, other: Tensor) -> Tensor:
        '''Returns the arc tangent^2 of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the arc tangent^2 operation.
        ''' 
        self._validate_other(other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        out_shape = self._broadcast_shape(self.shape, other.shape)
        if self.device == 'cuda': 
            out_data = cuda.math.atan2(other, self, out_shape)
        else: out_data = cpu.math.atan2(other, self)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=(self, other))
        
        def _backward() -> None:
            denom = (self**2 + other**2).clamp(min_value=1e-7)
            if self_requires_grad:
                grad = out.grad * other / denom
                self.grad += self._broadcast_grad(grad, self.shape)
            if other_requires_grad:
                grad = out.grad * (-self) / denom
                other.grad += self._broadcast_grad(grad, other.shape)
                
        out._backward = _backward
        return out
    
    ### SIGN ###
    
    def sign(self: Tensor) -> Tensor:
        '''Returns the sign of a tensor's data as new tensor.
        
        Returns:
            tensor : Resulting tensor from the sign operation.
        ''' 
                
        if self.device == 'cuda': out_data = cuda.math.sign(self)
        else: out_data = cpu.math.sign(self)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None: pass
        out._backward = _backward
        return out
    
    def copysign(self: Tensor, other: Tensor) -> Tensor:
        self._validate_other(other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        out_shape = self._broadcast_shape(self.shape, other.shape)
        
        if self.device == 'cuda': 
            out_data = cuda.math.copysign(self, other, out_shape)
        else: out_data = cpu.math.copysign(self, other)
        out = self._new(out_data, out_shape, self.dtype, self.device,
            requires_grad=self_requires_grad or other_requires_grad,
            _children=(self, other))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self._broadcast_grad(out.grad, self.shape)
            if other_requires_grad: 
                other.grad += self._broadcast_grad(out.grad, other.shape)
                
        out._backward = _backward
        return out
    
    ### REDUCTIONS ###
    
    def min(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        self_requires_grad = self.requires_grad

        if self.device == 'cuda':
            data = cuda.reductions.min(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.min(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = self._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                min_vals = out if keepdim else \
                        out.unsqueeze(dim) if dim is not None else \
                        out.reshape([1] * self.ndim)
                mask = (self == min_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward

        if dim is not None:
            _, indices = self.sort(dim=dim)
            idx = indices.select(dim, 0)
            if keepdim: idx = idx.unsqueeze(dim)
            return out, idx
        return out

    def amin(
        self: Tensor,
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self_requires_grad = self.requires_grad

        if isinstance(dim, (tuple, list)):
            ndim = self.ndim
            dims = sorted(set(d % ndim for d in dim), reverse=True)
            
            result = self
            for d in dims: result = result.amin(d, keepdim=True)
            
            if not keepdim:
                for d in dims: result = result.squeeze(d)
            return result

        if self.device == 'cuda':
            data = cuda.reductions.min(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.min(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = self._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                min_vals = out if keepdim else \
                    out.unsqueeze(dim) if dim is not None else \
                    out.reshape([1] * self.ndim)
                mask = (self == min_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward
        return out

    def max(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        self_requires_grad = self.requires_grad

        if self.device == 'cuda':
            data = cuda.reductions.max(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.max(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = self._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                max_vals = out if keepdim else \
                        out.unsqueeze(dim) if dim is not None else \
                        out.reshape([1] * self.ndim)
                mask = (self == max_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward

        if dim is not None:
            _, indices = self.sort(dim=dim, descending=True)
            idx = indices.select(dim, 0)
            if keepdim: idx = idx.unsqueeze(dim)
            return out, idx
        return out

    def amax(
        self: Tensor,
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self_requires_grad = self.requires_grad

        if isinstance(dim, (tuple, list)):
            ndim = self.ndim
            dims = sorted(set(d % ndim for d in dim), reverse=True)
            
            result = self
            for d in dims: result = result.amax(d, keepdim=True)
            
            if not keepdim:
                for d in dims: result = result.squeeze(d)
            return result

        if self.device == 'cuda':
            data = cuda.reductions.max(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.max(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = self._new(data, output_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                max_vals = out if keepdim else \
                        out.unsqueeze(dim) if dim is not None else \
                        out.reshape([1] * self.ndim)
                mask = (self == max_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)

        out._backward = _backward
        return out
    
    def argmin(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor:        
        data = np.argmin(self.cpu().numpy(), axis=dim)
        if keepdim and dim is not None:
            data = np.expand_dims(data, axis=dim)
        out = self._new(data, data.shape(), typing.int32, self.device)
            
        def _backward() -> None:
            raise RuntimeError(
                'argmin is not differentiable. Use amin() if you need '
                'gradients through a min operation.')

        out._backward = _backward
        return out

    def argmax(
        self: Tensor,
        dim: builtins.int | None = None,
        keepdim: bool = False
    ) -> Tensor:        
        data = np.argmax(self.cpu().numpy(), axis=dim)
        if keepdim and dim is not None:
            data = np.expand_dims(data, axis=dim)
        out = self._new(data, data.shape(), typing.int32, self.device)
        
        def _backward() -> None:
            raise RuntimeError(
                'argmargmaxin is not differentiable. Use amax() if you need '
                'gradients through a max operation.')

        out._backward = _backward
        return out
    
    def mean(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.mean(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.mean(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = cpu.reductions.mean(self, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = self._new(data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                n = self.size if dim is None else self.shape[dim]
                
                grad = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                
                self.grad += grad.expand(self.shape) / n
        
        out._backward = _backward
        return out
    
    @amp_float32
    def sum(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False,
        initial: builtins.int | builtins.float = 0.0
    ) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.sum(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.sum(self, dim, initial)
        else: data = cpu.reductions.sum(self, dim, keepdim, initial)

        output_shape = self.shape.reduce(dim, keepdim)
        out = self._new(data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad = out.grad if keepdim else out.grad.unsqueeze(dim) \
                    if dim is not None else out.grad.reshape([1] * self.ndim)
                self.grad += grad.expand(self.shape)
                
        out._backward = _backward
        return out
    
    @amp_float32
    def cumsum(self: Tensor, dim: builtins.int) -> Tensor:
        self_requires_grad = self.requires_grad

        dim = dim if dim >= 0 else self.ndim + dim

        if self.device == 'cuda':
            out_data = cuda.reductions.cumsum(self, dim)
        else: out_data = np.cumsum(self.data, axis=dim)

        out = self._new(out_data, self.shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = out.grad
                flipped = grad.flip(dim)
                flipped_csum = flipped.cumsum(dim)
                self.grad += flipped_csum.flip(dim)

        out._backward = _backward
        return out
        
    @amp_float32
    def prod(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False,
        initial: builtins.int | builtins.float = 1.0
    ) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.prod(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.prod(self, dim, initial)
        else: data = cpu.reductions.prod(self, dim, keepdim, initial)
        
        output_shape = self.shape.reduce(dim, keepdim)
        out = self._new(data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
            
        def _backward() -> None:
            if self_requires_grad:
                out_expanded = out if keepdim else out.unsqueeze(dim) \
                    if dim is not None else out.reshape([1] * self.ndim)
                grad_expanded = out.grad if keepdim else \
                    out.grad.unsqueeze(dim) if dim is not None else \
                    out.grad.reshape([1] * self.ndim)
                
                out_full = out_expanded.expand(self.shape)
                grad_full = grad_expanded.expand(self.shape)

                sign = (self >= 0.0).to(self.device, self.dtype) \
                     - (self <  0.0).to(self.device, self.dtype)
                safe_self = self.abs().clamp(min_value=1e-7) * sign
                self.grad += (out_full / safe_self) * grad_full
        
        out._backward = _backward
        return out

    def std(
        self: Tensor, 
        dim: builtins.int | tuple[builtins.int, ...] | None = None, 
        keepdim: bool = False, 
        correction: builtins.int = 1
    ) -> Tensor:
        mean = self.mean(dim=dim, keepdim=True)
        variance = ((self - mean) ** 2).mean(dim=dim, keepdim=keepdim)
        if correction == 1:
            n = self.size if dim is None else self.shape[dim]
            variance = variance * n / (n - 1)
        return variance.sqrt()
    
    @amp_float32
    def norm(
        self: Tensor,
        p: Literal['fro', 'l1', 'inf', '-inf', 'l0', 'lp'] = 'fro',
        dim: builtins.int | tuple[builtins.int, ...] | None = None,
        keepdim: bool = False
    ) -> Tensor:        
        match p:
            case 'fro': # L2/Frobenius norm
                return (self ** 2).sum(dim=dim, keepdim=keepdim).sqrt()
            case 'l1': # L1 norm
                return self.abs().sum(dim=dim, keepdim=keepdim)
            case 'inf': # L-(-inf) norm
                return self.abs().amax(dim=dim, keepdim=keepdim)
            case '-inf': # L-inf norm
                return self.abs().amin(dim=dim, keepdim=keepdim)
            case 'l0':  # L0 norm
                out = (self != 0).to(self.device, self.dtype)
                return out.sum(dim=dim, keepdim=keepdim)
            case _: # general Lp norm
                return (self.abs()**p).sum(dim=dim, keepdim=keepdim) ** (1.0/p)

    ### SORTING ###
    
    def sort(
        self: Tensor,
        dim: builtins.int = -1,
        descending: bool = False
    ) -> tuple[Tensor, Tensor]:
        self_requires_grad = self.requires_grad
        dim = dim if dim >= 0 else self.ndim + dim
        
        if self.device == 'cuda':
            out_data, indices = cuda.sorting.sort(self, dim, descending)
        else: out_data, indices = cpu.sorting.sort(self, dim, descending)
        
        indices = self._new(indices, self.shape, typing.int32, self.device)
        values  = self._new(out_data, self.shape, self.dtype, self.device,
            requires_grad=self_requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad_input = self._new(
                    np.zeros(self.grad.shape, self.grad.dtype),
                    self.grad.shape, self.grad.dtype, self.grad.device)
                grad_input = grad_input.scatter(dim, indices, values.grad)
                self.grad += grad_input
        
        values._backward = _backward
        return values, indices
    
    def argsort(
        self: Tensor,
        dim: builtins.int = -1,
        descending: bool = False
    ) -> Tensor:
        _, indices = self.sort(dim=dim, descending=descending)
        return indices
        
    ### INDEXING ###
    
    def scatter_add(
        self:   Tensor, 
        dim:    builtins.int, 
        index:  Tensor, 
        source: Tensor | builtins.int | builtins.float
    ) -> Tensor:
        assert self.shape == index.shape, \
            f'Shape of index tensor must match shape of input tensor.'
        
        if not isinstance(source, tensor):
            source = self._new(
                np.full(index.shape, fill_value=source, dtype=self.dtype), 
                index.shape, dtype=self.dtype, device=self.device)
            
        assert self.dtype == source.dtype, \
            f'Input and source must have the same dtype.'
        
        if not self.device == index.device or not self.device == source.device:
            _devices = set([self.device, index.device, source.device])
            raise ValueError(
                f'Scatter expects all tensors to be on same device, but found '
                f'multiple devices: {list(_devices)}')
        
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)

        self_requires_grad = self.requires_grad
        source_requires_grad = source.requires_grad
        
        if self.device == 'cuda':
            out_data = cuda.indexing.scatter_add(self, index, dim, source)
        else: out_data = cpu.indexing.scatter_add(
            self.data, index.data, dim, source.data)
        out = self._new(out_data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad
            if source_requires_grad: source.grad += out.grad.gather(dim, index)
        
        out._backward = _backward
        return out

    def masked_fill(
        self:  Tensor, 
        mask:  Tensor, 
        value: builtins.float
    ) -> Tensor:
        x = self * 0 + value
        return x * mask + self * (1 - mask)

