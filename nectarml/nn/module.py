from __future__ import annotations

from typing import Any, Literal

import numpy as np

from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, float32

class Module():
    _parameters: dict[str, Tensor]
    _submodules: dict[str, Module]
    _buffers:    dict[str, Tensor]
    
    def __init__(
        self: Module,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = float32
    ) -> None:
        super().__setattr__('_parameters', {})
        super().__setattr__('_submodules', {})
        super().__setattr__('_buffers', {})
        
        self.device = device
        self.dtype = dtype
        
        self.training:  bool = True
        self._device_id: int | None = None
                
    # PROPERTIES
    
    @property
    def device(self: Module) -> str:
        return self._device

    @device.setter
    def device(self: Module, value: str) -> None:
        self._device = value
        
    @property
    def dtype(self: Module) -> DTypeLike:
        return self._dtype
    
    @dtype.setter
    def dtype(self: Module, value: DTypeLike) -> None:
        self._dtype = value
        for parameter in self._parameters.values():
            parameter.dtype = value
            
    # REGISTRATION
    
    def register_parameter(self: Module, name: str, tensor: Tensor) -> None:
        self._parameters[name] = tensor
    
    def register_submodule(self: Module, name: str, module: Module) -> None:
        self._submodules[name] = module
        
    def register_buffer(self: Module, name: str, tensor: Tensor) -> None:
        self._buffers[name] = tensor
    
    # GETTERS / SETTERS
    
    def __setattr__(self: Module, name: str, value: Any) -> None:
        if '_buffers' in self.__dict__ and name in self._buffers:
            self._buffers[name] = value
        if isinstance(value, Module):
            self.register_submodule(name, value)
        elif isinstance(value, Tensor):
            self.register_parameter(name, value)
        else: super().__setattr__(name, value)
        
    def __getattr__(self, name: str) -> Any:
        if '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if '_submodules' in self.__dict__ and name in self._submodules:
            return self._submodules[name]
        else: raise AttributeError(
                f'{type(self).__name__} has no attribute "{name}"')
    
    # SUBMODULES
    
    def _walk_module_tree(self: Module) -> list[tuple[str, Module]]:
        tree: list[tuple[str, Module]] = []
        
        def walk_tree(prefix: str, node: Module):
            tree.append((prefix, node))
            for name, module in node._submodules.items():
                full_name = f'{prefix}.{name}' if prefix else name
                walk_tree(full_name, module)
            
        walk_tree('', self)
        return tree
    
    # GRADIENTS
    
    def zero_grad(self: Module) -> None:
        modules = self._walk_module_tree()
        for _, module in modules:
            for parameter in module._parameters.values():
                if not parameter.requires_grad: continue
                parameter.zero_grad()
    
    # DEVICE / DTYPE
    
    def to(
        self: Module,
        device: Literal['cpu', 'cuda'],
        dtype: DTypeLike | None = None
    ) -> Module: 
        modules = self._walk_module_tree()
        for _, module in modules:
            module.device = device
            if dtype is not None: module.dtype = dtype
            for buffer in module._buffers.values():  buffer.to(device, dtype)
            for parm in module._parameters.values(): parm.to(device, dtype)
       
        return self
    
    # STATES
    
    def train(self: Module) -> None: 
        modules = self._walk_module_tree()
        for _, module in modules:
            module.training = True
    
    def eval(self: Module) -> None:
        modules = self._walk_module_tree()
        for _, module in modules:
            module.training = False
        
    # INSPECTION
    
    def list_parameters(self: Module) -> list[tuple[str, Tensor]]:
        output = []
        modules = self._walk_module_tree()
        for module_name, module in modules:
            for parameter_name, parameter in module._parameters.items():
                name = f'{module_name}.{parameter_name}' if module_name \
                    else parameter_name
                output.append((name, parameter))
        return output
    
    def list_submodules(self: Module) -> list[tuple[str, Module]]:
        return [(name, module) for name, module in self._walk_module_tree()]
    
    def __repr__(self, indent: int = 0) -> str:
        pad = '  ' * indent
        child_pad = '  ' * (indent + 1)
        
        lines = [f'{type(self).__name__}(']
        
        for name, module in self._submodules.items():
            child_repr = module.__repr__(indent + 1)
            lines.append(f'{child_pad}({name}): {child_repr}')
        
        for name, param in self._parameters.items():
            parameter_line = (
                f'{child_pad}({name}): '
                f'Tensor(shape={param.shape}, dtype={param.dtype})')
            lines.append(parameter_line)
        
        lines.append(f'{pad})')
        
        return '\n'.join(lines)
        
    def __hash__(self: Module) -> int: return id(self)
    
    # FORWARD
    
    def forward(self: Module, *args, **kwargs): 
        raise NotImplementedError('forward() not implemented by child class.')

    def __call__(self: Module, *args, **kwargs):
        return self.forward(*args, **kwargs)



