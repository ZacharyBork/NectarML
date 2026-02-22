from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike

from nectarml import Tensor

class Module():
    _parameters: dict[str, Tensor]
    _submodules: dict[str, Module]
    
    def __init__(
        self,
        device: Literal['cpu', 'cuda'] = 'cpu',
        dtype: DTypeLike = np.float32
    ) -> None:
        super().__setattr__('_parameters', {})
        super().__setattr__('_submodules', {})
        
        self.device = device
        self.dtype = dtype
                
        self.training: bool = True
                
    # PROPERTIES
    
    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value
        
    @property
    def dtype(self) -> DTypeLike:
        return self._dtype
    
    @dtype.setter
    def dtype(self, value: DTypeLike) -> None:
        self._dtype = value
        for parameter in self._parameters.values():
            parameter.dtype = value
            
    # REGISTRATION
    
    def register_parameter(self, name: str, tensor: Tensor) -> None:
        self._parameters[name] = tensor
    
    def register_submodule(self, name: str, module: Module) -> None:
        self._submodules[name] = module
    
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self.register_submodule(name, value)
        elif isinstance(value, Tensor):
            self.register_parameter(name, value)
        else: super().__setattr__(name, value)
    
    # SUBMODULES
    
    def _walk_module_tree(self) -> list[tuple[str, Module]]:
        tree: list[tuple[str, Module]] = []
        
        def walk_tree(prefix: str, node: Module):
            tree.append((prefix, node))
            for name, module in node._submodules.items():
                full_name = f'{prefix}.{name}' if prefix else name
                walk_tree(full_name, module)
            
        walk_tree('', self)
        return tree
    
    # GRADIENTS
    
    def zero_grad(self) -> None:
        modules = self._walk_module_tree()
        for _, module in modules:
            for parameter in module._parameters.values():
                if not parameter.requires_grad: continue
                parameter.grad = np.zeros_like(parameter.data)
    
    # DEVICE / DTYPE
    
    def to(
        self,
        device: Literal['cpu', 'cuda'],
        dtype: DTypeLike | None = None
    ) -> Module: 
        modules = self._walk_module_tree()
        for _, module in modules:
            module.device = device
            if dtype is not None:
                module.dtype = dtype
        return self
    
    # STATES
    
    def train(self) -> None: 
        modules = self._walk_module_tree()
        for _, module in modules:
            module.training = True
    
    def eval(self) -> None:
        modules = self._walk_module_tree()
        for _, module in modules:
            module.training = False
        
    # INSPECTION
    
    def list_parameters(self) -> list[tuple[str, Tensor]]:
        output = []
        modules = self._walk_module_tree()
        for module_name, module in modules:
            for parameter_name, parameter in module._parameters.items():
                name = f'{module_name}.{parameter_name}' if module_name \
                    else parameter_name
                output.append((name, parameter))
        return output
    
    def list_submodules(self) -> list[tuple[str, Module]]:
        output = []
        modules = self._walk_module_tree()
        for name, module in modules:
            output.append((name, module))
        return output
    
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
        
    def __hash__(self) -> int: return id(self)
    
    # FORWARD
    
    def forward(self, *args, **kwargs): 
        raise NotImplementedError('forward() not implemented by child class.')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



