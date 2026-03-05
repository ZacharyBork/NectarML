from __future__ import annotations

from typing import Any, Literal
from collections.abc import Sequence, Callable

import numpy as np

from nectarml import typing
import nectarml.cuda as cuda
import nectarml._core as _core

class Tensor():
    def __init__(
        self,
        data: Any,
        shape: tuple[int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        self.device = device
        self._dtype = dtype
        
        self.data: np.ndarray | None = None
        self._data_ptr: int | None = None
        self._init_tensor(data, shape)       
        
        self.grad: np.ndarray | None = None
        self._grad_ptr: int | None = None
        self.requires_grad = requires_grad
        
        self._backward = lambda : None
        self._prev: set[Tensor] = set(_children)
    
    ### INIT ###
        
    def _init_tensor(
        self, 
        data: Any,
        shape: tuple[int, ...] | None = None
    ) -> None:
        if self.device == 'cpu':
            self.data = np.array(data, dtype=self.dtype)
            self.shape = shape or self.data.shape
        elif self.device == 'cuda': 
            if isinstance(data, int):
                if shape is None:
                    raise ValueError(
                        'Unable to init CUDA Tensor from device pointer '
                        'without explicit shape.')
            else: 
                array = np.array(data, dtype=self.dtype)
                shape = array.shape
                data = cuda.to_cuda(array, self.dtype)
            self._data_ptr = data
            self.shape = shape
        else: raise ValueError(f'Invalid device type: {self.device}')
    
    ### PROPERTIES ###
      
    @property
    def dtype(self) -> int:
        return self._dtype
        
    @property
    def cuda_dtype(self) -> int:
        return cuda.map_dtype(self.dtype)
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    @property
    def size(self) -> int:
        return np.prod(self.shape)
        
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if self.dtype != typing.bool_: 
            self._requires_grad = value
            if value:
                if self.device == 'cuda' and self._grad_ptr is None \
                or self.device == 'cpu' and self.grad is None:
                    self._allocate_grad()
            else: self._deallocate_grad()
        else: self._requires_grad = False
    
    ### DATA UTILS ###
    
    def numpy(self) -> np.ndarray:
        if self.device == 'cuda':
            return cuda.to_cpu(self._data_ptr, self.shape, self.dtype)
        return self.data
        
    ### GRADIENTS ###
    
    def _deallocate_grad(self) -> None:
        if self.device == 'cuda':
            if self._grad_ptr is not None:
                self._grad_ptr = cuda.free_cuda(self._grad_ptr)
        else: self.grad = None

    def _allocate_grad(self) -> None:
        if self.device == 'cuda':
            self._deallocate_grad()
            self._grad_ptr = cuda.alloc_cuda_full(
                self.size, cuda.map_dtype(typing.float32), 0.0)
        else: self.grad = np.zeros_like(self.data, dtype=typing.float32)
        
    def backward(self) -> None:
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
        self._allocate_grad()
        for node in graph: node._backward()
    
    def zero_grad(self) -> None:
        if self.requires_grad:
            self._allocate_grad()
            
    ### DEVICE / DTYPE ###
        
    def to(
        self,
        device: Literal['cpu', 'cuda'],
        dtype: typing.DTypeLike | None = None
    ) -> Tensor: 
        if device == self.device:
            if dtype is None or dtype == self.dtype:
                return self
                                
        dtype = dtype or self.dtype
        if device == 'cuda':
            shape = self.shape
            if self.device == 'cpu': data = cuda.to_cuda(self.data, dtype)
            else: data = cuda.cast_tensor(
                self._data_ptr, self.size, self.dtype, dtype)
        elif device == 'cpu':
            shape = None
            if self.device == 'cpu': data = self.data
            else: data = cuda.to_cpu(
                self._data_ptr, self.shape, self.dtype, dtype)
        else: raise ValueError(f'Invalid device type: {device}')
        
        new = Tensor(data=data, shape=shape, dtype=dtype, device=device, 
            requires_grad=self.requires_grad)
        new.grad = self.grad
        new._prev = self._prev
        new._backward = self._backward
        return new
        
    def cuda(self) -> Tensor: return self.to(device='cuda')

    def cpu(self) -> Tensor: return self.to(device='cpu')
    
    ### UTILS ###
    
    def _bool_type_check(self, op: str, other: Tensor | None = None) -> None:
        msg = f'Boolean tensors do not support operation: {op}'
        if isinstance(other, Tensor) and other.dtype == typing.bool_:
            raise RuntimeError(msg)
        if self.dtype == typing.bool_: raise RuntimeError(msg)
    
    def _validate_other(self, other: Tensor) -> None:
        assert isinstance(other, Tensor)
        assert self.device == other.device, (
            f'Expected all tensors to be on the same device, but found at '
            f'least two devices, {self.device} and {other.device}.')
        
    def _numerical_to_tensor(self, other: int | float) -> Tensor:
        new = Tensor(np.full(self.shape, other), dtype=self.dtype)
        return new.to(self.device)
    
    def _handle_tensor_or_numerical(
        self, 
        other: Tensor | int | float
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if isinstance(other, (float, int)):
            other = self._numerical_to_tensor(other)
            children = (self,)
        else: 
            self._validate_other(other)
            children = (self, other)
        return other, children
    
    def _build_output_tensor(
        self, 
        data: np.ndarray | int, 
        children: tuple[Tensor, ...]
    ) -> Tensor:
        _requires_grad = False
        for child in children:
            if child.requires_grad: _requires_grad = True
        return Tensor(
            data=data, shape=self.shape, device=self.device, 
            dtype=self.dtype, requires_grad=_requires_grad, _children=children)
    
    def _eval_core_function(
        self,
        func: Callable[
            [np.ndarray], 
            tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]]
    ) -> Tensor:
        out_data, _backward = func(self.data)
        out = self._build_output_tensor(out_data, (self,))
        def _backward_hook():
            if self.requires_grad:
                self.grad += _backward(out.grad)
        out._backward = _backward_hook
        return out
    
    ### REDUCTIONS ###
    
    def _get_reduce_shape(
        self, 
        dim: int | tuple[int, ...] | None, 
        keepdim: bool
    ) -> tuple[int, ...]:
        if dim is None: return (1,)
        
        out_shape = list(self.shape)
        if keepdim: out_shape[dim] = 1
        else: 
            if dim is not None:
                if isinstance(dim, (tuple, list)):
                    for idx, i in enumerate(dim): out_shape.pop(i-idx)
                else: out_shape.pop(dim)
        return tuple(out_shape)
    
    def min(
        self, 
        dim: int | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self._bool_type_check('Tensor.min()')
        _backward = lambda grad: grad
        
        if self.device == 'cuda':
            s = list(self.shape) if dim is not None else self.size
            data = cuda.reductions.min(self._data_ptr, s, dim, self.dtype)
            output_shape = self._get_reduce_shape(dim, keepdim)
        else:
            data, _backward = _core.reductions.min(
                self.data, dim, keepdim)
            output_shape = data.shape

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        out._backward = _backward
        return out
    
    def max(
        self, 
        dim: int | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self._bool_type_check('Tensor.max()')
        _backward = lambda grad: grad
        
        if self.device == 'cuda':
            s = list(self.shape) if dim is not None else self.size
            data = cuda.reductions.max(self._data_ptr, s, dim, self.dtype)
            output_shape = self._get_reduce_shape(dim, keepdim)
        else:
            data, _backward = _core.reductions.max(
                self.data, dim, keepdim)
            output_shape = data.shape

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        out._backward = _backward
        return out

    def argmin(
        self, 
        dim: int | None = None, 
        keepdim: bool = False
    ) -> typing.ArrayLike:
        self._bool_type_check('Tensor.argmin()')
        return _core.reductions.argmin(self.data, dim=dim, keepdim=keepdim)
        
    def argmax(
        self, 
        dim: int | None = None, 
        keepdim: bool = False
    ) -> typing.ArrayLike:
        self._bool_type_check('Tensor.argmax()')
        return _core.reductions.argmax(self.data, dim=dim, keepdim=keepdim)
    
    def mean(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        self._bool_type_check('Tensor.mean()')
        return self._eval_core_function(
            lambda x : _core.reductions.mean(x, dim=dim, keepdim=keepdim))
        
    def sum(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        initial: int | float = 0
    ) -> Tensor:
        self._bool_type_check('Tensor.sum()')
        _backward = lambda grad: grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.sum(d, keepdim=True)
                if not keepdim:
                    result.shape = result._get_reduce_shape(dim, keepdim)
                return result
            
            s = list(self.shape) if dim is not None else self.size
            data = cuda.reductions.sum(
                self._data_ptr, s, dim, self.dtype)
            output_shape = self._get_reduce_shape(dim, keepdim)
        else:
            data, _backward = _core.reductions.sum(
                self.data, dim, keepdim, initial)
            output_shape = data.shape

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        out._backward = _backward
        return out
    
    def prod(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        initial: int | float = 1
    ) -> Tensor:
        self._bool_type_check('Tensor.prod()')
        return self._eval_core_function(
            lambda x : _core.reductions.prod(x, dim, keepdim, initial))
        
    ### MATH OPS ###
    
    def abs(self) -> Tensor: 
        self._bool_type_check('Tensor.abs()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.abs(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.abs(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
            
    def exp(self) -> Tensor:
        self._bool_type_check('Tensor.exp()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.exp(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.exp(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
            
    def log(self) -> Tensor:
        self._bool_type_check('Tensor.log()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.log(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.log(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
            
    def sqrt(self) -> Tensor:
        self._bool_type_check('Tensor.sqrt()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.sqrt(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.sqrt(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
            
    def sin(self) -> Tensor:
        self._bool_type_check('Tensor.sin()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.sin(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.sin(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
        
    def cos(self) -> Tensor: 
        self._bool_type_check('Tensor.cos()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.cos(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.cos(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
        
    def tanh(self) -> Tensor:
        self._bool_type_check('Tensor.tanh()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.tanh(
                self._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.tanh(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward_hook():
            if self_requires_grad: self.grad += _backward(out.grad)

        out._backward = _backward_hook
        return out
        
    def sigmoid(self) -> Tensor: 
        self._bool_type_check('Tensor.sigmoid()')
        return ((-self).exp() + 1) ** -1
    
    ### RESHAPING ###
    
    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.reshape(x, shape))
        
    def flatten(self) -> Tensor:
        return self._eval_core_function(_core.shapes.flatten)
    
    def squeeze(self, dim: int | tuple[int, ...] | None) -> Tensor: 
        return self._eval_core_function(
            lambda x : _core.shapes.squeeze(x, dim))
    
    def unsqueeze(self, dim: int | tuple[int, ...]) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.unsqueeze(x, dim))
        
    def transpose(self, dims: Sequence[int] | None) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.transpose(x, dims))

    def swapdims(self, dim1: int, dim2: int) -> Tensor: 
        return self._eval_core_function(
            lambda x : _core.shapes.swapdims(x, dim1, dim2))
        
    def permute(self, dims: Sequence[int] | None) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.permute(x, dims))

    def expand(self, shape: tuple[int, ...]) -> Tensor:
        return self._eval_core_function(
            lambda x : _core.shapes.expand(x, shape))

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        return self.expand(shape)
        
    ### GETTERS / SETTERS ###
        
    def __getitem__(self, key: Any) -> int | float | bool:
        return self.numpy()[key]
    
    def __setitem__(self, key: Any, value: int | float | bool) -> None:
        self.data[key] = value
        
    def __str__(self) -> str: 
        data_str = np.array2string(
            self.numpy(), separator=', ', precision=4)
        data_str = data_str.replace('\n', '\n' + ' ' * 7)
        return f'Tensor({data_str}, device=\'{self.device}\')'
    
    def __repr__(self) -> str:
        return (
            f'shape: {self.shape}\n'
            f'data: {self.numpy()}\n'
            f'grad: {self.grad}\n'
            f'requires_grad: {self.requires_grad}\n'
            f'_prev: {self._prev}')
    
    def __len__(self) -> int: return self.numpy().__len__()
    
    def __hash__(self) -> int: return id(self)
    
    ### GARBAGE COLLECTION ###
    
    def __del__(self) -> None:
        if self.device == 'cuda' and self._data_ptr is not None:
            cuda.free_cuda(self._data_ptr)
        
    ### COMPARISON ###
    
    def __eq__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__eq__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.equal(
                self._data_ptr, other._data_ptr, self.size, self.dtype)
        else: data = self.data == other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __lt__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__lt__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.less_than(
                self._data_ptr, other._data_ptr, self.size, self.dtype)
        else: data = self.data < other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __le__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__le__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.less_than_or_equal(
                self._data_ptr, other._data_ptr, self.size, self.dtype)
        else: data = self.data <= other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __gt__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__gt__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.greater_than(
                self._data_ptr, other._data_ptr, self.size, self.dtype)
        else: data = self.data > other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __ge__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__ge__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.greater_than_or_equal(
                self._data_ptr, other._data_ptr, self.size, self.dtype)
        else: data = self.data >= other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    ### MATH OPS ###
    
    def __add__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__add__()', other)

        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.add(
                self._data_ptr, other._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.add(self.data, other.data)
        out = self._build_output_tensor(out_data, children)

        def _backward_hook():
            grad = _backward(out.grad)
            if self_requires_grad: self.grad += grad
            if other_requires_grad: other.grad += grad

        out._backward = _backward_hook
        return out
    
    def __radd__(self, other: Tensor | int | float) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__sub__()', other)
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad
            out_data = cuda.math.subtract(
                self._data_ptr, other._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.subtract(self.data, other.data)
        out = self._build_output_tensor(out_data, children)

        def _backward_hook():
            grad = _backward(out.grad)
            if self_requires_grad: self.grad += grad
            if other_requires_grad: other.grad += grad

        out._backward = _backward_hook
        return out
    
    def __rsub__(self, other: Tensor | int | float) -> Tensor:
        return (-self) + other
    
    def __neg__(self) -> Tensor:
        self._bool_type_check('Tensor.__neg__()')
        out_data, _backward = _core.math.negate(self.data)
        out = self._build_output_tensor(out_data, (self,))
        def _backward_hook():
            if self.requires_grad: self.grad += _backward(out.grad)
        out._backward = _backward_hook
        return out

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__sub__()', other)
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda':
            _backward = lambda grad: grad # NEEDS CUDA BACKPROP
            out_data = cuda.math.multiply(
                self._data_ptr, other._data_ptr, self.size, self.cuda_dtype)
        else: out_data, _backward = _core.math.multiply(self.data, other.data)
        out = self._build_output_tensor(out_data, children)

        def _backward_hook():
            grad = _backward(out.grad)
            if self_requires_grad: self.grad += grad
            if other_requires_grad: other.grad += grad

        out._backward = _backward_hook
        return out
    
    def __rmul__(self, other: Tensor | int | float) -> Tensor:
        return self * other
    
    def __matmul__(self, other: Tensor) -> Tensor:
        self._validate_other(other)
        self._bool_type_check('Tensor.__matmul__()', other)
        assert other.data.shape == self.data.shape
        if self.ndim == 1 or other.ndim == 1:
            raise NotImplementedError('matmul not supported for 1D tensors.')
        
        out_data, _backward = _core.math.matmul(self.data, other.data)
        out = self._build_output_tensor(out_data, (self, other))
        
        def _backward_hook():
            a_grad, b_grad = _backward(out.grad)
            if self.requires_grad: self.grad += a_grad
            if other.requires_grad: other.grad += b_grad
            
        out._backward = _backward_hook
        return out
    
    def __rmatmul__(self, other: Tensor) -> Tensor: return other @ self
    
    def __pow__(self, exponent: float | int) -> Tensor: 
        self._bool_type_check('Tensor.__pow__()')
        return self._eval_core_function(
            lambda x : _core.math.pow(x, exponent))
    
    def __rpow__(self, exponent: float | int) -> Tensor: 
        raise NotImplementedError
    
    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        self._bool_type_check('Tensor.__truediv__()', other)
        return self * other ** -1
    
    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        self._bool_type_check('Tensor.__rtruediv__()', other)
        return (self ** -1) * other
    
    def __abs__(self) -> Tensor:
        self._bool_type_check('Tensor.__abs__()')
        return self._eval_core_function(_core.math.abs)
    
    
    
