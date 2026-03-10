from __future__ import annotations

from typing import Any, Literal
from collections.abc import Sequence, Callable

import numpy as np

from nectarml import typing
import nectarml.cuda as cuda
import nectarml._core as _core

class CudaBuffer:
    def __init__(self, ptr: int, dtype: typing.DTypeLike) -> None:
        self.ptr = ptr
        self.dtype = dtype
        self._ref_count = 1
        
    def increment(self) -> CudaBuffer:
        self._ref_count += 1
        return self
        
    def decrement(self) -> None:
        self._ref_count -= 1
        if self._ref_count == 0:
            free = cuda.free_cuda
            if free is not None: free(self.ptr)

class Tensor():
    def __init__(
        self,
        data: Any,
        shape: typing.Size | tuple[int, ...] | None = None,
        dtype: typing.DTypeLike = typing.float32,
        device: Literal['cpu', 'cuda'] = 'cpu',
        requires_grad: bool = False,
        _children = ()
    ) -> None:
        self.device = device
        self._dtype = dtype
        self.requires_grad = requires_grad
        
        self.shape:         typing.Size = None
        self._device_id:     int | None = None
        self.data:    np.ndarray | None = None
        self._buffer: CudaBuffer | None = None
        self.grad:        Tensor | None = None
        
        self._backward: Callable = lambda : None
        self._prev:  set[Tensor] = set(_children)
        
        self._init_tensor(data, shape)       
        
    ### INIT ###
        
    def _init_tensor(
        self, 
        data: Any,
        shape: typing.Size | tuple[int, ...] | None = None
    ) -> None:
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
    def _data_ptr(self) -> int:
        return self._buffer.ptr
      
    @property
    def dtype(self) -> int:
        return self._dtype
    
    @property
    def ndim(self) -> int:
        return self.shape.ndim
    
    @property
    def size(self) -> int:
        return self.shape.numel()
        
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
        
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if self.dtype != typing.bool_: 
            self._requires_grad = value
        else: self._requires_grad = False
    
    ### DATA UTILS ###
    
    def numpy(self) -> np.ndarray:
        if self.device == 'cuda': 
            data = cuda.to_cpu(self)
            return data[0] if len(data) == 1 else data
        return self.data
    
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
        
    ### GRADIENTS ###
    
    def _deallocate_grad(self) -> None:
        self.grad = None

    def _allocate_grad(self) -> None:
        grad = np.zeros_like(self.data, dtype=typing.float32)
        if self.device == 'cuda':
            self._deallocate_grad()
            self.grad = Tensor(
                grad, self.shape, typing.float32, requires_grad=False
            ).to(self.device)
        else: self.grad = grad
        
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
        
    def cuda(self) -> Tensor: return self.to(device='cuda')

    def cpu(self) -> Tensor: return self.to(device='cpu')
    
    ### GETTERS / SETTERS ###
        
    def __getitem__(self, key: Any) -> int | float | bool:
        return self.numpy()[key]
    
    def __setitem__(self, key: Any, value: int | float | bool) -> None:
        self.data[key] = value
        
    def __str__(self) -> str: 
        data_str = np.array2string(
            self.numpy(), separator=', ', precision=4)
        data_str = data_str.replace('\n', '\n' + ' ' * 7)
        device_str = f'{self.device}' 
        if self.device == 'cuda' and self._device_id is not None:
            device_str = f'{device_str}:{self._device_id}'
        return f'Tensor({data_str}, device=\'{device_str}\')'
    
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
        if self.device == 'cuda' and self._buffer is not None:
            self._buffer.decrement()
    
    ### MASKING ###
    
    def mask(
        self, 
        value: Tensor | float, 
        op: Literal['eq', 'lt', 'le', 'gt', 'ge']
    ) -> Tensor:
        mod = cuda if self.device == 'cuda' else _core
        input = self if self.device == 'cuda' else self.data
        value = value.data if isinstance(value, Tensor) and not \
            self.device == 'cuda' else value
        
        match op:
            case 'eq': fn = mod.masking.eq_mask
            case 'lt': fn = mod.masking.lt_mask
            case 'le': fn = mod.masking.le_mask
            case 'gt': fn = mod.masking.gt_mask
            case 'ge': fn = mod.masking.ge_mask
            case _: raise ValueError(f'Invalid masking operation: {op}')
            
        return Tensor(fn(input, value), self.shape, self.dtype, self.device)
    
    ### COMPARISON ###
    
    def __eq__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__eq__()', other)
        self._validate_other(other)
        if self.device == 'cuda': data = cuda.math.equal(self, other)
        else: data = self.data == other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __lt__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__lt__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.less_than(self, other)
        else: data = self.data < other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __le__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__le__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.less_than_or_equal(self, other)
        else: data = self.data <= other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __gt__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__gt__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.greater_than(self, other)
        else: data = self.data > other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    def __ge__(self, other: Tensor) -> Tensor:
        self._bool_type_check('Tensor.__ge__()', other)
        self._validate_other(other)
        if self.device == 'cuda':
            data = cuda.math.greater_than_or_equal(self, other)
        else: data = self.data >= other.data
        return Tensor(data, self.shape, typing.bool_, self.device)
    
    ### MATH DUNDERS ###
    
    def __iadd__(self, other: Tensor | int | float) -> Tensor:
        other, _ = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.__add__()', other)

        if self.device == 'cuda': self._data_ptr = cuda.math.add(self, other)
        else: self.data += other.data
        return self
    
    def __add__(self, other: Tensor | int | float) -> Tensor:
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
    
    def __radd__(self, other: Tensor | int | float) -> Tensor:
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
    
    def __rsub__(self, other: Tensor | int | float) -> Tensor:
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

    def __mul__(self, other: Tensor | int | float) -> Tensor:
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
    
    def __rmul__(self, other: Tensor | int | float) -> Tensor:
        return self * other
    
    def __matmul__(self, other: Tensor) -> Tensor:
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
    
    def __rmatmul__(self, other: Tensor) -> Tensor: return other @ self
    
    def __pow__(self, exponent: float | int) -> Tensor: 
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
    
    def minimum(self, other: Tensor | float | int) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.minimum()', other)
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.minimum(self, other)
        else: out_data = _core.math.minimum(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self.mask(other, 'le') * out.grad
            if other_requires_grad:
                other.grad += other.mask(self, 'lt') * out.grad
        
        out._backward = _backward
        return out
        
    
    def maximum(self, other: Tensor | float | int) -> Tensor:
        other, children = self._handle_tensor_or_numerical(other)
        self._bool_type_check('Tensor.maximum()', other)
        
        self_requires_grad = self.requires_grad
        other_requires_grad = other.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.maximum(self, other)
        else: out_data = _core.math.maximum(self.data, other.data)
        out = self._build_output_tensor(out_data, children)
        
        def _backward() -> None:
            if self_requires_grad: 
                self.grad += self.mask(other, 'ge') * out.grad
            if other_requires_grad:
                other.grad += other.mask(self, 'gt') * out.grad
        
        out._backward = _backward
        return out
    
    def clamp(
        self, 
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
                mask = self.scalar_mask(min_value, 'ge') \
                     * self.scalar_mask(max_value, 'le')
                self.grad += mask * out.grad
        
        out._backward = lambda : _backward(out.grad)
        return out
    
    ### ABS ###

    def abs(self) -> Tensor: return self.__abs__()
        
    ### EXP ###
            
    def exp(self) -> Tensor:
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
            
    def log(self) -> Tensor:
        self._bool_type_check('Tensor.log()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log(self)
        else: out_data = _core.math.log(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / self) * out.grad

        out._backward = _backward
        return out
    
    def log2(self) -> Tensor:
        self._bool_type_check('Tensor.log2()') 
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda': out_data = cuda.math.log2(self)
        else: out_data = _core.math.log2(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += out.grad / (self * np.log(2))

        out._backward = _backward
        return out
    
    def log10(self) -> Tensor:
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
            
    def sqrt(self) -> Tensor:
        self._bool_type_check('Tensor.sqrt()')
        self_requires_grad = self.requires_grad
        
        if self.device == 'cuda':out_data = cuda.math.sqrt(self)
        else: out_data = _core.math.sqrt(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None:
            if self_requires_grad: self.grad += (1 / (2 * out)) * out.grad

        out._backward = _backward
        return out
    
    def rsqrt(self) -> Tensor:
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
            
    def sin(self) -> Tensor:
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
    
    def asin(self) -> Tensor:
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
    
    def sinh(self) -> Tensor:
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
    
    def asinh(self) -> Tensor:
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
        
    def cos(self) -> Tensor: 
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
    
    def acos(self) -> Tensor:
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
    
    def cosh(self) -> Tensor:
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
    
    def acosh(self) -> Tensor:
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
    
    def tan(self) -> Tensor:
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
        
    def tanh(self) -> Tensor:
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
    
    def atan(self) -> Tensor:
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
    
    def atanh(self) -> Tensor:
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
    
    def atan2(self, other: Tensor) -> Tensor:
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
    
    def sign(self) -> Tensor:
        self._bool_type_check('Tensor.sign()')
                
        if self.device == 'cuda': out_data = cuda.math.sign(self)
        else: out_data = _core.math.sign(self.data)
        out = self._build_output_tensor(out_data, (self,))

        def _backward() -> None: pass
        out._backward = _backward
        return out
       
    ### SIGMOID ###
        
    def sigmoid(self) -> Tensor: 
        self._bool_type_check('Tensor.sigmoid()')
        return ((-self).exp() + 1) ** -1
    
    ### REDUCTIONS ###
    
    def min(
        self, 
        dim: int | None = None,
        keepdim: bool = False
    ) -> Tensor:
        self._bool_type_check('Tensor.min()')
        _backward = lambda grad: grad
        
        if self.device == 'cuda':
            data = cuda.reductions.min(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data, _backward = _core.reductions.min(
                self.data, dim, keepdim)
            output_shape = typing.Size(data.shape)

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
            data = cuda.reductions.max(self, dim)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data, _backward = _core.reductions.max(
                self.data, dim, keepdim)
            output_shape = typing.Size(data.shape)

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
        _backward = lambda grad: grad
        
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
            data, _backward = _core.reductions.mean(
                self.data, dim, keepdim)
            output_shape = typing.Size(data.shape)

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        out._backward = _backward
        return out
    
    def sum(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        initial: int | float = 0.0
    ) -> Tensor:
        self._bool_type_check('Tensor.sum()')
        _backward = lambda grad: grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.sum(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.sum(self, dim, initial)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data, _backward = _core.reductions.sum(
                self.data, dim, keepdim, initial)
            output_shape = typing.Size(data.shape)

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        out._backward = _backward
        return out
    
    def prod(
        self, 
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        initial: int | float = 1.0
    ) -> Tensor:
        self._bool_type_check('Tensor.prod()')
        _backward = lambda grad: grad
        
        if self.device == 'cuda':
            if isinstance(dim, (tuple, list)):
                result = self
                for d in sorted(dim, reverse=True):
                    result = result.sum(d, keepdim=True)
                if not keepdim:
                    result.shape = result.shape.reduce(dim, keepdim)
                return result
            
            data = cuda.reductions.prod(self, dim, initial)
            output_shape = self.shape.reduce(dim, keepdim)
        else:
            data, _backward = _core.reductions.prod(
                self.data, dim, keepdim, initial)
            output_shape = typing.Size(data.shape)

        out = Tensor(
            data, output_shape, self.dtype, self.device, self.requires_grad)
        out._backward = _backward
        return out

    ### RESHAPING ###
    
    def reshape(self, shape: tuple[int, ...]) -> Tensor:
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
        
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        end_dim = end_dim if end_dim >= 0 else self.ndim + end_dim
        new_shape = (
            self.shape[:start_dim]
          + (int(np.prod(self.shape[start_dim:end_dim+1])),)
          + self.shape[end_dim+1:])
        return self.reshape(new_shape)
        
    def squeeze(self, dim: int | None = None) -> Tensor:
        if dim is None: new_shape = tuple(s for s in self.shape if s != 1)
        else:
            if self.shape[dim] != 1: return self 
            new_shape = self.shape[:dim] + self.shape[dim+1:]
        return self.reshape(new_shape)

    def unsqueeze(self, dim: int) -> Tensor:
        dim = dim if dim >= 0 else self.ndim + dim + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(new_shape)
            
    def permute(self, dims: tuple[int, ...] | None) -> Tensor:
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

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(tuple(dims))

    def swapdims(self, dim1: int, dim2: int) -> Tensor: 
        return self.transpose(dim1, dim2)

    def expand(self, shape: tuple[int, ...]) -> Tensor:
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

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        return self.expand(shape)
    
    ### COMBINATION ###
    
    def concatenate(self, inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
        inputs = [self] + inputs
                
        if self.device == 'cuda':
            _backward = lambda grad : grad
            data = cuda.combinations.concatenate(inputs, dim)
            shape = self.shape
            requires_grad = self.requires_grad
            for i in inputs[1:]: 
                shape[dim] += list(i.shape)[dim]
                if i.requires_grad: requires_grad = True
        else:
            data, _backward = _core.combination.concatenate(
                [t.data for t in inputs], dim=dim)
            shape = typing.Size(data.shape)
            requires_grad = self.requires_grad
        
        out = Tensor(data, shape, self.dtype, self.device, 
            requires_grad, tuple(inputs))

        def _backward_hook():
            grads = _backward(out.grad)
            for tensor, grad in zip(inputs, grads):
                if tensor.requires_grad:
                    tensor.grad += grad
        
        out._backward = _backward_hook
        return out
    
    def cat(self, inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
        return self.concatenate(inputs, dim)
    
