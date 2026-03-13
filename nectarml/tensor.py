from __future__ import annotations

import warnings
from typing import Any, Literal
from collections.abc import Callable

import numpy as np

from nectarml import typing
from nectarml.cuda.memory import CudaBuffer
import nectarml.cuda as cuda
import nectarml._core as _core

class Tensor():
    def __init__(
        self: Tensor,
        data: Any,
        shape: typing.Size | tuple[int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        self.device = device
        self._dtype = dtype
        
        self.shape:         typing.Size = None
        self._device_id:     int | None = None
        self.data:    np.ndarray | None = None
        self._buffer: CudaBuffer | None = None
        self.grad:        Tensor | None = None
        
        self._backward: Callable = lambda : None
        self._prev:  set[Tensor] = set(_children)
        
        self._init_tensor(data, shape) 
        if requires_grad: self._allocate_grad()      
        self.requires_grad = requires_grad
        
    ### INIT ###
        
    def _init_tensor(
        self: Tensor, 
        data: Any,
        shape: typing.Size | tuple[int, ...] | None = None
    ) -> None:
        '''Initializes a Tensor from given data and optional shape.
        
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
                
        Raises:
            ValueError : If data is uintptr to CUDA tensor data and shape is
                not provided.
            ValueError : If Tensor's device type is not valid (i.e. not "cpu"
                or "cuda").
        '''
        if self.device == 'cpu':
            self.data = np.array(data, dtype=self.dtype)
            self.shape = shape if isinstance(shape, typing.Size) else \
                typing.Size(shape or self.data.shape)
        elif self.device == 'cuda': 
            self._device_id = 0 # NEEDS TO BE UPDATED FOR REAL MULTI-GPU ID
            if isinstance(data, int):
                if shape is None:
                    raise ValueError(
                        'Unable to init CUDA Tensor from device pointer '
                        'without explicit shape.')
                self._buffer = CudaBuffer(data, self.dtype)
                self.shape = shape if isinstance(shape, typing.Size) \
                    else typing.Size(shape)
            else: 
                self.data = np.array(data, dtype=self.dtype)
                self.shape = shape if isinstance(shape, typing.Size) else \
                    typing.Size(shape or self.data.shape)
                self._buffer = CudaBuffer(cuda.to_cuda(self), self.dtype)
        else: raise ValueError(f'Invalid device type: {self.device}')
    
    ### PROPERTIES ###
      
    @property
    def _data_ptr(self: Tensor) -> int | None:
        '''Property access for CUDA pointer to Tensor's data.
        
        Returns:
            int | None : If the given Tensor owns a CudaBuffer (i.e. if its
                device is "cuda"), returns the uintptr to the Tensor's data in
                CUDA memory. Otherwise returns NoneType.
        '''
        if self._buffer is not None: return self._buffer.ptr
        return None
      
    @property
    def dtype(self: Tensor) -> typing.DTypeLike:
        '''Property access for Tensor's Dtype.
        
        Returns:
            typing.DtypeLike : The Dtype of the Tensor.
        '''
        return self._dtype
    
    @property
    def ndim(self: Tensor) -> int:
        '''Property access for number of dims in Tensor.
        
        Returns:
            int : The number of dimensions in the given Tensor's shape.
        '''
        return self.shape.ndim
    
    @property
    def size(self: Tensor) -> int:
        '''Propert access for tensor size (i.e. numel).
        
        Returns:
            int : The number of elements in the given number shape. Equivelent
                to math.prod(Tensor.shape).
        '''
        return self.shape.numel()
        
    @property
    def requires_grad(self: Tensor) -> bool:
        '''Property access for Tensor's requires_grad value.
        
        Returns:
            bool : True if given Tensor requires grad, otherwise False.
        '''
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self: Tensor, value: bool) -> None:
        '''Setter for Tensor's requires_grad value.
        
        This setter will allocate a grad tensor for the given data Tensor if
        value=True and the Tensor's grad is None. If value=False, it will set
        the given Tensor's grad to None.
        
        Args:
            value : True to enable grad on the given Tensor, False to disable.
        '''
        if self.dtype != typing.bool_: 
            self._requires_grad = value
            if value and self.grad is None: self._allocate_grad()
        else: 
            self._requires_grad = False
            self._deallocate_grad()
    
    ### DATA UTILS ###
    
    def numpy(self: Tensor) -> np.ndarray:
        '''Returns the Tensor's data as a numpy.ndarray.
        
        Returns:
            np.ndarray : The Tensor's data as a numpy.ndarray.
        '''
        if self.device == 'cuda': 
            data = cuda.to_cpu(self)
            return data[0] if len(data) == 1 else data
        return self.data
    
    def tolist(self: Tensor) -> list[Any]:
        return self.numpy().tolist()
    
    def item(self: Tensor) -> int | float:
        return self.numpy().item()
    
    def is_floating_point(self: Tensor) -> bool:
        return self.dtype in [typing.float, typing.float16, typing.float32]
    
    def is_cuda(self: Tensor) -> bool:
        return self.device == 'cuda'
    
    def is_cpu(self: Tensor) -> bool:
        return self.device == 'cpu'
    
    def dim(self: Tensor) -> int:
        return self.shape.ndim
    
    def numel(self: Tensor) -> int:
        return self.shape.numel()
    
    ### UTILS ###
    
    def detach(self: Tensor) -> Tensor:
        if self.device == 'cuda':
            out = Tensor(self._data_ptr, self.shape, self.dtype, self.device)
            out._buffer = self._buffer.increment()
        else: out = Tensor(self.data, self.shape, self.dtype, self.device)
        return out
    
    def detach_(self: Tensor) -> None:
        self.requires_grad = False
        self._backward = None
        self._prev.clear()
    
    def clone(self: Tensor) -> Tensor:
        if self.device == 'cuda':
            clone_ptr = cuda.clone(self)
            out = Tensor(clone_ptr, self.shape, self.dtype, self.device,
                self.requires_grad, _children=(self,))
        else: out = Tensor(self.data.copy(), self.shape, self.dtype, 
                self.device, self.requires_grad, _children=(self,))
        
        self_requires_grad = self.requires_grad
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad
        
        out._backward = _backward
        return out
    
    def requires_grad_(self: Tensor, value: bool) -> Tensor:
        self.requires_grad = value
        return self
    
    def fill_(self: Tensor, fill_value: float | int) -> Tensor:
        if self.device == 'cuda':
            new_ptr = cuda.memory.alloc_cuda_full(
                self.size, self.dtype, fill_value)
            old_buffer = self._buffer
            self._buffer = CudaBuffer(new_ptr, self.dtype)
            old_buffer.decrement()
        else: self.data.fill(fill_value)
        return self
    
    def zero_(self: Tensor) -> Tensor:
        return self.fill_(0.0)
    
    def copy_(self: Tensor, other: Tensor) -> Tensor:
        assert self.shape == other.shape, \
            f'copy_ requires Tensors to have the same shape.'
        assert self.dtype == other.dtype, \
            f'copy_ requires Tensors to have the same dtype.'
        if self.device == 'cuda':
            if other.device == 'cuda': copy_ptr = cuda.utils.clone(other)
            else: copy_ptr = cuda.data_to_cuda(
                other.data, other.size, other.dtype)
            self._buffer.decrement()
            self._buffer = CudaBuffer(copy_ptr, self.dtype)
        else: np.copyto(self.data, other.numpy())
        return self
    
    def cuda_build_shape_(self: Tensor) -> None:
        self.shape = typing.Size(self.numpy().shape)
    
    def _bool_type_check(
        self: Tensor, 
        op: str, 
        other: Tensor | None = None
    ) -> None:
        '''Ensures self and optional "other" Tensor are not boolean Tensors.
        
        This is used as a guard on operations that do not allow boolean Tensors
        to validate and print a clean error message if not valid.
        
        Args:
            op : The name of the parent operation which is running the check.
                Used to build error string for printing if self or other is
                not valid.
            other : Optional other Tensor to check for bool type. Used for ops
                which require more than one Tensor (i.e. __add__, __mul__).
                
        Raises:
            RuntimeError : If self (or other if present) is bool type Tensor.
        '''
        msg = f'Boolean tensors do not support operation: {op}'
        if isinstance(other, Tensor) and other.dtype == typing.bool_:
            raise RuntimeError(msg)
        if self.dtype == typing.bool_: raise RuntimeError(msg)
    
    def _validate_other(self: Tensor, other: Tensor) -> None:
        assert isinstance(other, Tensor)
        assert self.device == other.device, (
            f'Expected all tensors to be on the same device, but found at '
            f'least two devices, {self.device} and {other.device}.')
    
    def _handle_tensor_or_numerical(
        self: Tensor, 
        other: Tensor | int | float
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        '''Helper to handle conversion for ops that allow numerical data.
        
        If "other" is numerical data (float | int), a new tensor will be
        created from the given data. The new tensor will match the shape of the 
        Tensor this is called from, and will have the same device and dtype.
        
        If "other" is a Tensor object, the Tensor will be validated and passed
        through as is.
        
        Args:
            other : The other Tensor object to check, or numerical data to 
                convert to new Tensor.
        Returns:
            tuple[Tensor, tuple[Tensor, ...]] : A tuple containing the new (or
                validated) other Tensor, and a tuple of Tensors containing
                the Tensor this is called from, and the other Tensor, to be
                used as children for backpropagation.
        '''
        if isinstance(other, (float, int)):
            other = Tensor(
                np.full(self.shape, other), dtype=self.dtype
            ).to(self.device)
            children = (self,)
        else: 
            self._validate_other(other)
            children = (self, other)
        return other, children
    
    def _build_output_tensor(
        self: Tensor, 
        data: np.ndarray | int, 
        children: tuple[Tensor, ...]
    ) -> Tensor:
        '''Helper method to build output Tensor's from Tensor ops.
        
        Args:
            data : Either ArrayLike data for new Tensor, or uintptr to new
                Tensor's data in CUDA memory.
            children : The child Tensors to assign to the newly created Tensor
                for backpropagation.
        '''
        _requires_grad = False
        for child in children:
            if child.requires_grad: _requires_grad = True
        return Tensor(
            data=data, shape=self.shape, device=self.device, 
            dtype=self.dtype, requires_grad=_requires_grad, _children=children)
        
    ### GRADIENTS ###
    
    def _deallocate_grad(self: Tensor) -> None:
        '''Deallocated grad by setting Tensor.grad to None.'''
        self.grad = None

    def _allocate_grad(self: Tensor, fill_value: float = 0.0) -> None:
        '''Allocates gradient tensor for given tensor.
        
        Args:
            fill_value : The value to fill the new grad Tensor with.
        '''
        self._deallocate_grad()
        self.grad = Tensor(np.full(self.shape, fill_value, typing.float32), 
            self.shape, typing.float32, self.device, requires_grad=False) 
        if self.device == 'cuda': self.grad = self.grad.cuda()
        
    def backward(self: Tensor) -> None:
        '''Gradient backpropagation method.
        
        Builds a topo graph from all children of the Tensor it was called on
        recursively. Then allocates a new gradient Tensor for the given Tensor,
        fills the gradient tensor with ones, and then walks the graph, calling
        each subsequent child's _backward() method.
        '''
        assert self.ndim == 0 or self.size == 1, \
            'backward() can only be called on scalar tensors.'
        
        visited: set[int] = set()
        graph: list[Tensor] = []
        
        def build_graph(node: Tensor):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_graph(child)
                graph.append(node)
        
        build_graph(self)
        graph.reverse()
        self._allocate_grad(fill_value=1.0)
        for node in graph: node._backward()
    
    def zero_grad(self: Tensor) -> None:
        '''Zeros values in the grad Tensor of the Tensor it is called on.'''
        if self.requires_grad:
            self._allocate_grad()
            
    ### DEVICE / DTYPE ###
        
    def to(
        self: Tensor,
        device: Literal['cpu', 'cuda'],
        dtype: typing.DTypeLike | None = None
    ) -> Tensor: 
        '''Casts tensor to new device and/or Dtype.
        
        Args:
            device : The device to cast the Tensor to ["cpu", "cuda"].
            dtype : The Dtype to cast the Tensor to.
            
        Returns:
            Tensor : The resulting Tensor from the cast operation.
        '''
        if device == self.device:
            if dtype is None or dtype == self.dtype:
                return self
                                
        dtype = dtype or self.dtype
        if device == 'cuda':
            shape = self.shape
            if self.device == 'cpu': data = cuda.to_cuda(self)
            else: data = cuda.cast_tensor(self, dtype)
        elif device == 'cpu':
            shape = None
            if self.device == 'cpu': data = self.data
            else: data = cuda.to_cpu(self, dtype)
        else: raise ValueError(f'Invalid device type: {device}')
        
        new = Tensor(data=data, shape=shape, dtype=dtype, device=device, 
            requires_grad=self.requires_grad)
        new.grad = self.grad.to(self.device) if self.grad is not None else None
        new._prev = self._prev
        new._backward = self._backward
        return new
        
    def cuda(self: Tensor) -> Tensor: 
        '''Convenience function to cast given Tensor to CUDA device.
        
        Returns:
            Tensor : The resulting CUDA Tensor from the cast operation.
        '''
        return self.to(device='cuda')

    def cpu(self: Tensor) -> Tensor: 
        '''Convenience function to cast given Tensor to CPU device.
        
        Returns:
            Tensor : The resulting CPU Tensor from the cast operation.
        '''
        return self.to(device='cpu')
    
    ### GETTERS / SETTERS ###
        
    def __getitem__(self: Tensor, idx: int | slice | tuple[slice]) -> Tensor:
        self_requires_grad = self.requires_grad
        
        if not isinstance(idx, tuple): idx = (idx,)
        
        idx = list(idx)
        while len(idx) < self.ndim: idx.append(slice(None))
        
        squeeze_dims = []
        normalized: list[slice] = []
        for i, index in enumerate(idx):
            if isinstance(index, int):
                if index < 0: index = self.shape[i] + index
                normalized.append(slice(index, index + 1, 1))
                squeeze_dims.append(i)
            elif isinstance(index, slice): normalized.append(index)
            else: raise ValueError(f'Index type not valid: {type(index)}')
        
        if self.device == 'cuda':
            out_data = cuda.indexing.slice_tensor(self, tuple(normalized))
        else: out_data = self.data[tuple(normalized)]
        
        out_shape = []
        for i, s in enumerate(normalized):
            start = s.start if s.start is not None else 0
            stop  = s.stop  if s.stop  is not None else self.shape[i]
            step  = s.step  if s.step  is not None else 1
            
            if start < 0: start = self.shape[i] + start
            if stop < 0:  stop  = self.shape[i] + stop
            
            start = max(0, min(self.shape[i], start))
            stop  = max(0, min(self.shape[i], stop))
            
            out_shape.append((stop - start + step - 1) // step)
        
        out_shape = typing.Size(out_shape)
        out = Tensor(out_data, out_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        for dim in reversed(squeeze_dims): out = out.squeeze(dim)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = Tensor(np.zeros(self.shape, typing.float32), self.shape, 
                    typing.float32, self.device, requires_grad=False)
                grad[tuple(normalized)] = out.grad
                self.grad += grad
        
        out._backward = _backward
        return out
    
    def __setitem__(
        self: Tensor, 
        idx: int | slice | tuple[slice],
        value: Any
    ) -> None:
        warnings.warn(
            'Tensor.__setitem__() modifies the given Tensor in-place. This '
            'operation does not support autograd. Calling this method on a '
            'Tensor in the autograd graph will corrupt the gradients of any '
            'Tensor which depends on it.')
        
        if self.device == 'cuda':
            new_ptr = cuda.indexing.index_put(self, idx, value)
            self._buffer = CudaBuffer(new_ptr, self.dtype)
        else:
            if isinstance(value, Tensor): value = value.data
            self.data[idx] = value
        
    def __str__(self: Tensor) -> str: 
        '''Returns Tensor info (data, device) as a formatted string.
        
        Returns:
            str : The Tensor info string.
        '''
        data_str = np.array2string(
            self.numpy(), separator=', ', precision=4)
        data_str = data_str.replace('\n', '\n' + ' ' * 7)
        device_str = f'{self.device}' 
        if self.device == 'cuda' and self._device_id is not None:
            device_str = f'{device_str}:{self._device_id}'
        return f'Tensor({data_str}, device=\'{device_str}\')'
    
    def __repr__(self: Tensor) -> str:
        return (
            f'shape: {self.shape}\n'
            f'data: {self.numpy()}\n'
            f'grad: {self.grad}\n'
            f'requires_grad: {self.requires_grad}\n'
            f'_prev: {self._prev}')
    
    def __len__(self) -> int: 
        '''Gets the length (in elements) of the Tensor's data.
        
        Returns:
            int : The Tensor's length (in elements).
        '''
        return self.numpy().__len__()
    
    def __hash__(self: Tensor) -> int: 
        '''Hash override. Returns memory address of Tensor.
        
        Returns:
            int : Address in system memory of the Tensor.
        '''
        return id(self)
    
    ### GARBAGE COLLECTION ###
    
    def __del__(self: Tensor) -> None:
        '''Tensor garbage collection.
        
        Used to decrement the CudaBuffer of the Tensor if device="cuda". If
        CudaBuffer reference count reaches zero when this Tensor is deleted,
        the CUDA memory of the Tensor is cleared and the garbage collector
        is then allowed to clean up the CudaBuffer object.
        '''
        if self.device == 'cuda' and self._buffer is not None:
            self._buffer.decrement()
    
    ### COMPARISON ###
    
    def __eq__(self: Tensor, other: Tensor | int | float) -> Tensor:
        '''Tensor elementwise equality operator.
        
        Args:
            other : The Tensor, int, or float value to compare the given Tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new Tensor with the same shape, device,
                and Dtype as the Tensor this operator is called on.
                
        Returns:
            Tensor : A boolean Tensor denoting whether each element is equal
                to the corresponding value in the "other" Tensor. Note: This
                return Tensor can be turned into a binary mask like so:
                
                    x = Tensor(data, dtype=typing.float32)
                    y = Tensor(data, dtype=typing.float32)
                    mask = (x == y).to(x.device, x.dtype)
        '''
        self._bool_type_check('Tensor.__eq__()', other)
        other, _ = self._handle_tensor_or_numerical(other)
        if self.device == 'cuda': data = cuda.math.equal(self, other)
        else: data = self.data == other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __lt__(self: Tensor, other: Tensor | int | float) -> Tensor:
        '''Tensor elementwise less than operator.
        
        Args:
            other : The Tensor, int, or float value to compare the given Tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new Tensor with the same shape, device,
                and Dtype as the Tensor this operator is called on.
                
        Returns:
            Tensor : A boolean Tensor denoting whether each element is less 
                than to the corresponding value in the "other" Tensor. Note: 
                This return Tensor can be turned into a binary mask like so:
                
                    x = Tensor(data, dtype=typing.float32)
                    y = Tensor(data, dtype=typing.float32)
                    mask = (x < y).to(x.device, x.dtype)
        '''
        self._bool_type_check('Tensor.__lt__()', other)
        other, _ = self._handle_tensor_or_numerical(other)
        if self.device == 'cuda':
            data = cuda.math.less_than(self, other)
        else: data = self.data < other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __le__(self: Tensor, other: Tensor | int | float) -> Tensor:
        '''Tensor elementwise less than or equal operator.
        
        Args:
            other : The Tensor, int, or float value to compare the given Tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new Tensor with the same shape, device,
                and Dtype as the Tensor this operator is called on.
                
        Returns:
            Tensor : A boolean Tensor denoting whether each element is less 
                than or equal to the corresponding value in the "other" Tensor. 
                Note: This return Tensor can be turned into a binary mask 
                like so:
                
                    x = Tensor(data, dtype=typing.float32)
                    y = Tensor(data, dtype=typing.float32)
                    mask = (x <= y).to(x.device, x.dtype)
        '''
        self._bool_type_check('Tensor.__le__()', other)
        other, _ = self._handle_tensor_or_numerical(other)
        if self.device == 'cuda':
            data = cuda.math.less_than_or_equal(self, other)
        else: data = self.data <= other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __gt__(self: Tensor, other: Tensor | int | float) -> Tensor:
        '''Tensor elementwise greater than operator.
        
        Args:
            other : The Tensor, int, or float value to compare the given Tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new Tensor with the same shape, device,
                and Dtype as the Tensor this operator is called on.
                
        Returns:
            Tensor : A boolean Tensor denoting whether each element is greater 
                than to the corresponding value in the "other" Tensor. Note: 
                This return Tensor can be turned into a binary mask like so:
                
                    x = Tensor(data, dtype=typing.float32)
                    y = Tensor(data, dtype=typing.float32)
                    mask = (x > y).to(x.device, x.dtype)
        '''
        self._bool_type_check('Tensor.__gt__()', other)
        other, _ = self._handle_tensor_or_numerical(other)
        if self.device == 'cuda':
            data = cuda.math.greater_than(self, other)
        else: data = self.data > other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __ge__(self: Tensor, other: Tensor | int | float) -> Tensor:
        '''Tensor elementwise greater than or equal operator.
        
        Args:
            other : The Tensor, int, or float value to compare the given Tensor
                against. Note: If other is an integer or float, the value will
                first be used to fill a new Tensor with the same shape, device,
                and Dtype as the Tensor this operator is called on.
                
        Returns:
            Tensor : A boolean Tensor denoting whether each element is greater 
                than or equal to the corresponding value in the "other" Tensor. 
                Note: This return Tensor can be turned into a binary mask 
                like so:
                
                    x = Tensor(data, dtype=typing.float32)
                    y = Tensor(data, dtype=typing.float32)
                    mask = (x >= y).to(x.device, x.dtype)
        '''
        self._bool_type_check('Tensor.__ge__()', other)
        other, _ = self._handle_tensor_or_numerical(other)
        if self.device == 'cuda':
            data = cuda.math.greater_than_or_equal(self, other)
        else: data = self.data >= other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    ### MATH DUNDERS ###
    
    def __iadd__(self: Tensor, other: Tensor | int | float) -> Tensor:
        other, _ = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__add__()', other)

        if self.device == 'cuda': self._data_ptr = cuda.math.add(self, other)
        else: self.data += other.data
        return self
    
    def __add__(self: Tensor, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__add__()', other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.add(self, other)
        else: out_data = _core.math.add(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad
            if other_requires_grad: other.grad += out.grad
                
        out._backward = lambda : _backward(out.grad)
        return out
    
    def __radd__(self: Tensor, other: Tensor | int | float) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__sub__()', other)
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.subtract(self, other)   
        else: out_data = _core.math.subtract(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad
            if other_requires_grad: other.grad += out.grad

        out._backward = lambda : _backward(out.grad)
        return out
    
    def __rsub__(self: Tensor, other: Tensor | int | float) -> Tensor:
        return (-self) + other
    
    def __neg__(self) -> Tensor:
        self._bool_type_check('Tensor.__neg__()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.negate(self)
        else: out_data = _core.math.negate(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += -out.grad

        out._backward = _backward
        return out

    def __mul__(self: Tensor, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__sub__()', other)
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.multiply(self, other)
        else: out_data = _core.math.multiply(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad: self.grad += other.grad * out.grad
            if other_requires_grad: other.grad += self.grad * out.grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    def __rmul__(self: Tensor, other: Tensor | int | float) -> Tensor:
        return self * other
    
    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        self._validate_other(other)
        self._bool_type_check('Tensor.__matmul__()', other)
        if self.ndim == 1 or other.ndim == 1:
            raise NotImplementedError('matmul not supported for 1D tensors.')
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.matmul(self, other)
        else: out_data = _core.math.matmul(self.data, other.data)
        out = self._build_output_tensor(out_data, (self, other))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += cuda.math.matmul(
                    out.grad, other.transpose(-2, -1))
            if other_requires_grad:
                other.grad += cuda.math.matmul(
                    self.transpose(-2, -1), out.grad)
        
        out._backward = _backward
        return out
    
    def __rmatmul__(self: Tensor, other: Tensor) -> Tensor: return other @ self
    
    def __pow__(self: Tensor, exponent: float | int) -> Tensor: 
        self._bool_type_check('Tensor.__pow__()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.pow(self, exponent)
        else: out_data = _core.math.pow(self.data, exponent)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += exponent * (self**(exponent-1)) * out.grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    def __rpow__(self: Tensor, exponent: float | int) -> Tensor: 
        raise NotImplementedError
    
    def __truediv__(self: Tensor, other: Tensor | float | int) -> Tensor:
        self._bool_type_check('Tensor.__truediv__()', other)
        return self * other ** -1
    
    def __rtruediv__(self: Tensor, other: Tensor | float | int) -> Tensor:
        self._bool_type_check('Tensor.__rtruediv__()', other)
        return (self ** -1) * other
    
    def __abs__(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__abs__()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.abs(self)
        else: out_data = _core.math.abs(self.data)
        
        def _backward(out_grad: Tensor) -> None:
            if self_requires_grad: 
                self.grad += self.sign() * out_grad
        
        out = self._build_output_tensor(out_data, (self,))
        out._backward = lambda : _backward(out.grad)
        return out
    
    ### CLAMP ###
    
    def minimum(self: Tensor, other: Tensor | float | int) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.minimum()', other)
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.minimum(self, other)
        else: out_data = _core.math.minimum(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = (self < other).to(self.device, self.dtype) 
                self.grad += grad * out.grad
            if other_requires_grad:
                grad = (other < self).to(other.device, other.dtype)
                other.grad += grad * out.grad
        
        out._backward = _backward
        return out
        
    
    def maximum(self: Tensor, other: Tensor | float | int) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.maximum()', other)
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.maximum(self, other)
        else: out_data = _core.math.maximum(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad:
                grad = (self >= other).to(self.device, self.dtype) 
                self.grad += grad * out.grad
            if other_requires_grad:
                grad = (other <= self).to(other.device, other.dtype)
                other.grad += grad * out.grad
        
        out._backward = _backward
        return out
    
    def clamp(
        self: Tensor, 
        min_value: float | None = None, 
        max_value: float | None = None
    ) -> Tensor:
        self._bool_type_check('Tensor.clamp()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': 
            out_data = cuda.math.clamp(self, min_value, max_value)
        else: out_data = _core.math.clamp(self.data, min_value, max_value)
        out = self._build_output_tensor(out_data, (self,))
                    
        def _backward() -> None:
            if self_requires_grad:
                mask = (self >= min_value).to(self.device, self.dtype) \
                     * (self <= max_value).to(self.device, self.dtype)
                self.grad += mask * out.grad
        
        out._backward = _backward
        return out
    
    ### ABS ###

    def abs(self: Tensor) -> Tensor: return self.__abs__()
        
    ### EXP ###
            
    def exp(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.exp()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.exp(self)
        else:  out_data = _core.math.exp(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out * out.grad

        out._backward = _backward
        return out
      
    ### LOG ###
            
    def log(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.log()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log(self)
        else: out_data = _core.math.log(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / self) * out.grad

        out._backward = _backward
        return out
    
    def log2(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.log2()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log2(self)
        else: out_data = _core.math.log2(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(2))

        out._backward = _backward
        return out
    
    def log10(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.log10()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log10(self)
        else: out_data = _core.math.log10(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(10))

        out._backward = _backward
        return out
          
    ### SQRT ###
            
    def sqrt(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.sqrt()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.sqrt(self)
        else: out_data = _core.math.sqrt(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / (2 * out)) * out.grad

        out._backward = _backward
        return out
    
    def rsqrt(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.rsqrt()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.rsqrt(self)
        else: out_data = _core.math.rsqrt(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += -0.5 * out**3 * out.grad

        out._backward = _backward
        return out
    
    ### SIN / COS ###
            
    def sin(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.sin()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.sin(self)
        else: out_data = _core.math.sin(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            # NOTE: This builds unneeded graph nodes. Eventually, this should
            # be replaced with something akin to torch.no_grad()
            if self_requires_grad: self.grad += self.cos() * out.grad

        out._backward = _backward
        return out
    
    def asin(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.asin()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.asin(self)
        else: out_data = _core.math.asin(self.data)
        
        def _backward() -> None:
            if self_requires_grad:
                denom = (1 - self ** 2).sqrt().clamp(min_value=1e-7)
                self.grad += out.grad / denom
        
        out = self._build_output_tensor(out_data, (self,))
        out._backward = lambda : _backward(out.grad)
        return out
    
    def sinh(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.sinh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.sinh(self)
        else: out_data = _core.math.sinh(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += self.cosh() * out.grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    def asinh(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.asinh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.asinh(self)
        else: out_data = _core.math.asinh(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad / (self**2 - 1).sqrt()
        
        out._backward = lambda : _backward(out.grad)
        return out
        
    def cos(self: Tensor) -> Tensor: 
        self._bool_type_check('Tensor.cos()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.cos(self)
        else: out_data = _core.math.cos(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            # NOTE: This builds unneeded graph nodes. Eventually, this should
            # be replaced with something akin to torch.no_grad()
            if self_requires_grad: self.grad += -self.sin() * out.grad

        out._backward = _backward
        return out
    
    def acos(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.acos()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.acos(self)
        else: out_data = _core.math.acos(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = (1 - self**2).sqrt().clamp(min_value=1e-7)
                self.grad += -out.grad / grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    def cosh(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.cosh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.cosh(self)
        else: out_data = _core.math.cosh(self.data)
        out = self._build_output_tensor(out_data, (self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += self.sinh() * out.grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    def acosh(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.acosh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.acosh(self)
        else: out_data = _core.math.acosh(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                grad = (self**2 - 1).sqrt().clamp(min_value=1e-7)
                self.grad += out.grad / grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    ### TAN / ATAN ###
    
    def tan(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.tan()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.tan(self)
        else: out_data = _core.math.tan(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad:
                self.grad += (1 + out**2) * out.grad
        
        out._backward = lambda : _backward(out.grad)
        return out
        
    def tanh(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.tanh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.tanh(self)
        else: out_data = _core.math.tanh(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += (1 - out**2) * out.grad

        out._backward = _backward
        return out
    
    def atan(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.atan()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atan(self)
        else: out_data = _core.math.atan(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += out.grad / (1 + self**2)

        out._backward = _backward
        return out
    
    def atanh(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.atanh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atanh(self)
        else: out_data = _core.math.atanh(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward():
            if self_requires_grad: 
                self.grad += out.grad / (1 - self**2).clamp(min_value=1e-7)

        out._backward = _backward
        return out
    
    def atan2(self: Tensor, other: Tensor) -> Tensor:
        self._validate_other(other)
        self._bool_type_check('Tensor.__add__()', other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.atan2(other, self)
        else: out_data = _core.math.atan2(other.data, self.data)
        out = self._build_output_tensor(out_data, (self, other))
        
        def _backward() -> None:
            denom = None
            if self_requires_grad: 
                denom = (self**2 + out**2).clamp(min_value=1e-7)
                self.grad += out.grad * -out / denom
            if other_requires_grad: 
                denom = denom or (self**2 + out**2).clamp(min_value=1e-7)
                other.grad += out.grad * self / denom
                
        out._backward = _backward
        return out
    
    ### SIGN ###
    
    def sign(self: Tensor) -> Tensor:
        self._bool_type_check('Tensor.sign()')
                
        if self.device == 'cuda': out_data = cuda.math.sign(self)
        else: out_data = _core.math.sign(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None: pass
        out._backward = _backward
        return out
       
    ### SIGMOID ###
        
    def sigmoid(self: Tensor) -> Tensor: 
        self._bool_type_check('Tensor.sigmoid()')
        return ((-self).exp() + 1) ** -1
    
    ### REDUCTIONS ###
    
    def min(
        self: Tensor, 
        dim: int | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self._bool_type_check('Tensor.min()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': data = cuda.reductions.min(self, dim)
        else: data = _core.reductions.min(self.data, dim, keepdim)
        output_shape = self.shape.reduce(dim, keepdim)
        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad,
            _children=(self,))
            
        def _backward() -> None:
            if self_requires_grad:
                min_vals = out if keepdim else out.unsqueeze(dim) \
                    if dim is not None else out.reshape([1] * self.ndim)
                mask = (self == min_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else out.grad.unsqueeze(dim) \
                    if dim is not None else out.grad.reshape([1] * self.ndim)
                self.grad += mask * grad.expand(self.shape)
        
        out._backward = _backward
        return out
    
    def max(
        self: Tensor, 
        dim: int | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self._bool_type_check('Tensor.max()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            data = cuda.reductions.max(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data = _core.reductions.max(
                self.data, dim, keepdim)
            output_shape = typing.Size(data.shape)
        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad,
            _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                max_vals = out if keepdim else out.unsqueeze(dim) \
                    if dim is not None else out.reshape([1] * self.ndim)
                mask = (self == max_vals.expand(self.shape)).to(
                    self.device, self.dtype)
                grad = out.grad if keepdim else out.grad.unsqueeze(dim) \
                    if dim is not None else out.grad.reshape([1] * self.ndim)
                self.grad += mask *  grad.expand(self.shape)
        
        out._backward = _backward
        return out

    def argmin(
        self: Tensor, 
        dim: int | None = None, 
        keepdim: bool = False
    ) -> typing.ArrayLike:
        self._bool_type_check('Tensor.argmin()')
        if self.device == 'cuda':
            raise RuntimeError(
                'argmin currently not supported for CUDA tensors.')
        return _core.reductions.argmin(self.data, dim=dim, keepdim=keepdim)
        
    def argmax(
        self: Tensor, 
        dim: int | None = None, 
        keepdim: bool = False
    ) -> typing.ArrayLike:
        self._bool_type_check('Tensor.argmax()')
        if self.device == 'cuda':
            raise RuntimeError(
                'argmax currently not supported for CUDA tensors.')
        return _core.reductions.argmax(self.data, dim=dim, keepdim=keepdim)
    
    def mean(
        self: Tensor, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        self._bool_type_check('Tensor.mean()')
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
            data = _core.reductions.mean(
                self.data, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        
        def _backward() -> None:
            if self_requires_grad:
                n = 1
                if dim is None: n = self.size
                else: n = self.shape[dim]
                
                grad = out.grad if keepdim else out.grad.unsqueeze(dim) \
                    if dim is not None else out.grad.reshape([1] * self.ndim)
                self.grad += grad.expand(self.shape) / n
        
        out._backward = _backward
        return out
    
    def sum(
        self: Tensor, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        initial: int | float = 0.0
    ) -> Tensor:
        self._bool_type_check('Tensor.sum()')
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
        else: data = _core.reductions.sum(self.data, dim, keepdim, initial)

        output_shape = self.shape.reduce(dim, keepdim)
        out = Tensor(
            data, output_shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad = out.grad if keepdim else out.grad.unsqueeze(dim) \
                    if dim is not None else out.grad.reshape([1] * self.ndim)
                self.grad += grad.expand(self.shape)
                
        out._backward = _backward
        return out
    
    def prod(
        self: Tensor, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        initial: int | float = 1.0
    ) -> Tensor:
        self._bool_type_check('Tensor.prod()')
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
        else: data = _core.reductions.prod(self.data, dim, keepdim, initial)
        
        output_shape = self.shape.reduce(dim, keepdim)
        out = Tensor(
            data, output_shape, self.dtype, self.device, 
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

    ### RESHAPING ###
    
    def reshape(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        self_requires_grad = self.requires_grad
        orig_shape = self.shape
        
        if self.device == 'cuda':
            out = Tensor(
                self._data_ptr, shape, self.dtype, 
                self.device, self.requires_grad, _children=(self,))
            self._buffer = out._buffer.increment()
        else:
            out = Tensor(
                _core.shapes.reshape(self.data, shape), shape, self.dtype, 
                self.device, self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad.reshape(orig_shape)
                
        out._backward = _backward
        return out
        
    def flatten(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        end_dim = end_dim if end_dim >= 0 else self.ndim + end_dim
        new_shape = (
            self.shape[:start_dim]
          + (int(np.prod(self.shape[start_dim:end_dim+1])),)
          + self.shape[end_dim+1:])
        return self.reshape(new_shape)
        
    def squeeze(self: Tensor, dim: int | None = None) -> Tensor:
        if dim is None: new_shape = tuple(s for s in self.shape if s != 1)
        else:
            if self.shape[dim] != 1: return self 
            new_shape = self.shape[:dim] + self.shape[dim+1:]
        return self.reshape(new_shape)

    def unsqueeze(self: Tensor, dim: int) -> Tensor:
        dim = dim if dim >= 0 else self.ndim + dim + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(new_shape)
            
    def permute(self: Tensor, dims: tuple[int, ...] | None) -> Tensor:
        self_requires_grad = self.requires_grad
    
        if self.device == 'cuda': out_data = cuda.shapes.permute(self, dims)
        else: out_data = _core.shapes.permute(self.data, dims) 
        out_shape = tuple(self.shape[d] for d in dims)
        out = Tensor(out_data, out_shape, self.dtype, self.device,
            self.requires_grad, _children=(self,))
        
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims): inv_dims[d] = i
        inv_dims = tuple(inv_dims)
        
        def _backward() -> None:
            if self_requires_grad:
                self.grad += out.grad.permute(inv_dims)
        
        out._backward = _backward
        return out

    def transpose(self: Tensor, dim1: int, dim2: int) -> Tensor:
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(tuple(dims))

    def swapdims(self: Tensor, dim1: int, dim2: int) -> Tensor: 
        return self.transpose(dim1, dim2)

    def expand(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        assert len(shape) == self.ndim, \
            f'expand target shape must have same ndim as input'
        orig_shape = self.shape
        assert all(t == s or s == 1 for s, t in zip(orig_shape, shape)), \
            f'expand can only expand size-1 dimensions'
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.shape.expand(self, shape)
        else: out_data = _core.shapes.expand(self.data, shape).copy()
        out = Tensor(out_data, shape, self.dtype, self.device,
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

    def broadcast_to(self: Tensor, shape: tuple[int, ...]) -> Tensor:
        return self.expand(shape)
    
    ### COMBINATION ###
    
    def select(self: Tensor, dim: int, index: int) -> Tensor:
        idx = [index if i == dim else slice(None) for i in range(self.ndim)]
        return self[tuple(idx)]

    def unstack(self: Tensor, dim: int = 0) -> list[Tensor]:
        return [self.select(dim, i) for i in range(self.shape[dim])]
        
    def unbind(self: Tensor, dim: int = 0) -> list[Tensor]:
        return self.unstack(dim)
        
    def split(
        self: Tensor, 
        split_size: int | list[int], 
        dim: int = 0
    ) -> list[Tensor]:
        if isinstance(split_size, int):
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

    def chunk(self: Tensor, size: int, dim: int = 0) -> list[Tensor]:
        assert size >= 1
        chunk_size = int(np.ceil(self.shape[dim] / size))
        return self.split(chunk_size, dim)
    
    ### INDEXING ###
    
    def gather(self: Tensor, dim: int | None, index: Tensor) -> Tensor:
        assert index.device == self.device, (
            f'Gather expects input Tensor and index Tensor to be on same '
            f'device, but found two devices, {self.device} and {index.device}')
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)
        
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            out_data = cuda.indexing.gather(self, dim, index)
        else: out_data = _core.indexing.gather(self.data, dim, index.data)
        out = Tensor(out_data, index.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad:
                grad = Tensor(
                    np.zeros(self.shape, typing.float32), 
                    self.shape, typing.float32, self.device, 
                    requires_grad=False)
                self.grad += grad.scatter_add(dim, index, out.grad)
        
        out._backward = _backward
        return out

    def scatter(
        self: Tensor, 
        dim: int, 
        index: Tensor, 
        source: Tensor | int | float
    ) -> Tensor:
        if not isinstance(source, Tensor):
            source = Tensor(np.full(index.shape, fill_value=source), 
                dtype=self.dtype, device=self.device)
        
        if not self.device == index.device or not self.device == source.device:
            _devices = set([self.device, index.device, source.device])
            raise ValueError(
                f'Scatter expects all Tensors to be on same device, but found '
                f'multiple devices: {list(_devices)}')
        assert self.dtype == source.dtype, \
            f'Input and source must have the same dtype.'
        
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)
            
        self_requires_grad = self.requires_grad
        source_requires_grad = source.requires_grad
        
        if self.device == 'cuda':
            out_data = cuda.indexing.scatter(self, index, dim, source)
        else: out_data = _core.indexing.scatter(
            self.data, index.data, dim, source.data)
        out = Tensor(out_data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))

        def _backward() -> None:
            if source_requires_grad:
                source.grad += out.grad.gather(dim, index)
            if self_requires_grad:
                mask = Tensor(
                    np.ones(self.shape, typing.float32),
                    self.shape, typing.float32, self.device,
                    requires_grad=False)
                mask = mask.scatter(dim, index, 0.0)
                self.grad += out.grad * mask             
            
        out._backward = _backward
        return out
    
    def scatter_add(
        self: Tensor, 
        dim: int, 
        index: Tensor, 
        source: Tensor | int | float
    ) -> Tensor:
        assert self.shape == index.shape, \
            f'Shape of index tensor must match shape of input tensor.'
        
        if not isinstance(source, Tensor):
            source = Tensor(
                np.full(index.shape, fill_value=source, dtype=self.dtype), 
                dtype=self.dtype, device=self.device)
            
        assert self.dtype == source.dtype, \
            f'Input and source must have the same dtype.'
        
        if not self.device == index.device or not self.device == source.device:
            _devices = set([self.device, index.device, source.device])
            raise ValueError(
                f'Scatter expects all Tensors to be on same device, but found '
                f'multiple devices: {list(_devices)}')
        
        if index.dtype != typing.int32:
            index = index.to(index.device, dtype=typing.int32)

        self_requires_grad = self.requires_grad
        source_requires_grad = source.requires_grad
        
        if self.device == 'cuda':
            out_data = cuda.indexing.scatter_add(self, index, dim, source)
        else: out_data = _core.indexing.scatter_add(
            self.data, index.data, dim, source.data)
        out = Tensor(out_data, self.shape, self.dtype, self.device, 
            self.requires_grad, _children=(self,))
        
        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad
            if source_requires_grad: source.grad += out.grad.gather(dim, index)
        
        out._backward = _backward
        return out

    def masked_fill(self: Tensor, mask: Tensor, value: float) -> Tensor:
        x = self * 0 + value
        return x * mask + self * (1 - mask)

    def index_select(self: Tensor, dim: int, index: Tensor) -> int:
        idx_shape = [1] * self.ndim
        idx_shape[dim] = len(index)
        index = index.reshape(tuple(idx_shape))
        
        gather_shape = list(self.shape)
        gather_shape[dim] = index.shape[dim]
        index = index.expand(tuple(gather_shape))
        
        return self.gather(dim, index)
    
    
