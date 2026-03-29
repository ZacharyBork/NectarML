from __future__ import annotations

from typing import Any
from collections import OrderedDict
from collections.abc import Iterable, ItemsView, KeysView, ValuesView

from nectarml.nn.module import Module

class ModuleDict(Module):
    def __init__(
        self: ModuleDict,
        modules: dict[str, Any]
    ) -> None:
        super().__init__()
        self.modules_dict = modules
    
    def clear(self: ModuleDict) -> None:
        self.modules_dict.clear()
    
    def items(self: ModuleDict) -> ItemsView[str, Module]:
        return self.modules_dict.items()
    
    def keys(self: ModuleDict) -> KeysView[str]:
        return self.modules_dict.keys()
    
    def values(self: ModuleDict) -> ValuesView[Module]:
        return self.modules_dict.values()
    
    def pop(self: ModuleDict, key: str) -> Module:
        return self.modules_dict.pop(key)
    
    def update(self: ModuleDict, modules: dict[str, Module]) -> None:
        self.modules_dict.update(modules)

    def _walk_module_tree(self) -> list[tuple[str, Module]]:
        result = [('', self)]
        for name, module in self.modules_dict.items():
            if isinstance(module, Module):
                for subname, submodule in module._walk_module_tree():
                    full_name = f'{name}.{subname}' if subname else name
                    result.append((full_name, submodule))
        return result

    def __getitem__(self: ModuleDict, key: str) -> Module:
        return self.modules_dict[key]

    def __setitem__(self: ModuleDict, key: str, module: Module) -> None:
        self.modules_dict[key] = module

    def __delitem__(self: ModuleDict, key: str) -> None:
        del self.modules_dict[key]

    def __contains__(self: ModuleDict, key: str) -> bool:
        return key in self.modules_dict

    def __len__(self: ModuleDict) -> int:
        return len(self.modules_dict)

class ModuleList(Module):
    def __init__(
        self: ModuleList, 
        modules: list[Module]
    ) -> None:
        super().__init__()
        self.modules_list = list(modules)

    def append(self: ModuleList, module: Module) -> None:
        self.modules_list.append(module)
    
    def extend(self: ModuleList, other: ModuleList) -> None:
        self.modules_list.extend(other.modules_list)
    
    def insert(self: ModuleList, index: int, module: Module) -> None:
        self.modules_list.insert(index, module)  
        
    def _walk_module_tree(self) -> list[tuple[str, Module]]:
        result = [('', self)]
        for i, module in enumerate(self.modules_list):
            if isinstance(module, Module):
                for subname, submodule in module._walk_module_tree():
                    full_name = f'{i}.{subname}' if subname else str(i)
                    result.append((full_name, submodule))
        return result
        
    def __getitem__(self: ModuleList, index: int) -> Module:
        return self.modules_list[index]

    def __setitem__(self: ModuleList, index: int, module: Module) -> None:
        self.modules_list[index] = module

    def __delitem__(self: ModuleList, index: int) -> None:
        del self.modules_list[index]

    def __len__(self: ModuleList) -> int:
        return len(self.modules_list)

    def __iter__(self: ModuleList):
        return iter(self.modules_list)

    def __contains__(self: ModuleList, module: Module) -> bool:
        return module in self.modules_list

class Sequential(ModuleList):
    def __init__(
        self: Sequential, 
        *modules: Module | Iterable[Module] | OrderedDict[str, Module]
    ) -> None:
        if len(modules) == 1:
            if isinstance(modules[0], Module): modules = [modules]
            if isinstance(modules[0], OrderedDict):
                modules = list(modules[0].values())
            if isinstance(modules[0], list | tuple):
                modules = list(modules[0])
        else: modules = list(modules)
        super().__init__(modules)
    
    def pop(self: Sequential, index: int) -> Module:
        return self.modules_list.pop(index)
    
    def forward(self: Sequential, *inputs) -> Any:
        if not self.modules_list:
            return inputs[0] if len(inputs) == 1 else inputs
        out = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            out = module(out)
        return out
    

