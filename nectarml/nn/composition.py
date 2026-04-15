from __future__ import annotations

from typing import Any
from collections import OrderedDict
from collections.abc import Iterable, Iterator, ItemsView, KeysView, ValuesView

from nectarml.tensor import Tensor
from nectarml.nn.module import Module

class ModuleDict(Module):
    def __init__(
        self:    ModuleDict,
        modules: dict[str, Module]
    ) -> None:
        '''A dictionary to store modules with string lookup keys.
        
        Args:
            modules : A dictionary object containing the Modules to store
                with string keys.
        '''
        super().__init__()
        self.modules_dict = modules
    
    def clear(self: ModuleDict) -> None:
        '''Clears the ModuleDict's internal Module dictionary.'''
        self.modules_dict.clear()
    
    def items(self: ModuleDict) -> ItemsView[str, Module]:
        '''Returns a view on the ModuleDicts items.
        
        Returns:
            ItemsView : A view on the items of the ModuleDict instance.
        '''
        return self.modules_dict.items()
    
    def keys(self: ModuleDict) -> KeysView[str]:
        '''Returns a view on the ModuleDicts keys.
        
        Returns:
            KeysView : A view on the keys of the ModuleDict instance.
        '''
        return self.modules_dict.keys()
    
    def values(self: ModuleDict) -> ValuesView[Module]:
        '''Returns a view on the ModuleDicts values.
        
        Returns:
            ValuesView : A view on the values of the ModuleDict instance.
        '''
        return self.modules_dict.values()
    
    def pop(self: ModuleDict, key: str) -> Module:
        '''Pops the item with the given key from the ModuleDict and returns it.
        
        Args:
            key : The string key of the item to pop from the ModuleDict.
        
        Returns:
            Module : The Module popped from the ModuleDict.
            
        Raises:
            KeyError : If the given key is not found in the ModuleDict.
        '''
        return self.modules_dict.pop(key)
    
    def update(self: ModuleDict, modules: dict[str, Module]) -> None:
        '''Updates the ModuleDict with the input modules dictionary.
        
        Args:
            modules : The dictionary of string mapped Modules to update the
                ModuleDict with.
        '''
        self.modules_dict.update(modules)

    def _walk_module_tree(self: ModuleDict) -> list[tuple[str, Module]]:
        '''Walks the full Module tree of the ModuleDict and returns all items.
        
        This method recurses each module contained within the ModuleDict to
        return a list containing not just the modules directly owned by the
        ModuleDict, but also all of their submodules.
        
        Returns:
            list[tuple[str, Module]] : A list of every Module owned by the
                ModuleDict (and all of their submodules), as 
                tuple[str, Module] where the Module is a reference to the 
                Module itself, and the string is the Module's full name.
                
        '''
        result = [('', self)]
        for name, module in self.modules_dict.items():
            if isinstance(module, Module):
                for subname, submodule in module._walk_module_tree():
                    full_name = f'{name}.{subname}' if subname else name
                    result.append((full_name, submodule))
        return result

    def __getitem__(self: ModuleDict, key: str) -> Module:
        '''Gets a reference to the Module with the given key and returns it.
        
        Args:
            key : The key of the Module to get a reference to.
            
        Returns:
            Module : The Module at the given key.
            
        Raises:
            KeyError : If the given key is not present in the ModuleDict.
        '''
        return self.modules_dict[key]

    def __setitem__(self: ModuleDict, key: str, module: Module) -> None:
        '''Sets the given key in the ModuleDict to the provided item.
        
        If the given key is present in the ModuleDict, the Module it points to
        will be overridden with the provided Module. If the key does not exist,
        it will be created and the give Module will be assigned to it.
        
        Args:
            key : The key to set in the ModuleDict.
            Module : The module to assign to the given key.
        '''
        self.modules_dict[key] = module

    def __delitem__(self: ModuleDict, key: str) -> None:
        '''Removed the item with the given key from the ModuleDict.
        
        Args:
            key : The key of the item to delete from the ModuleDict.
            
        Raises:
            KeyError : If the given key is not present in the ModuleDict.
        '''
        del self.modules_dict[key]

    def __contains__(self: ModuleDict, key: str) -> bool:
        '''Checks whether a given key is present in the ModuleDict.
        
        Args:
            key : The key to check for.
            
        Returns:
            bool : True if the key is present in the ModuleDict, otherwise
                False.
        '''
        return key in self.modules_dict

    def __len__(self: ModuleDict) -> int:
        '''Returns the length of the ModuleDict as an integer.

        Returns:
            int : The length of the ModuleDict.
        '''
        return len(self.modules_dict)

class ModuleList(Module):
    def __init__(
        self:    ModuleList, 
        modules: list[Module]
    ) -> None:
        '''A list-type object to store Modules.
        
        Args:
            modules : The list of Modules to store in the ModuleList instance.
        '''
        super().__init__()
        self.modules_list = list(modules)
    
    def append(self: ModuleList, module: Module) -> None:
        '''Appends a new Module to the ModuleList.
        
        Args:
            module : The Module to append to the ModuleList.
        '''
        self.modules_list.append(module)
    
    def extend(self: ModuleList, other: ModuleList) -> None:
        '''Extends the ModuleList with the items from another ModuleList.
        
        Args:
            other : The ModuleList to extend the given ModuleList with.
        '''
        self.modules_list.extend(other.modules_list)
    
    def insert(self: ModuleList, index: int, module: Module) -> None:
        '''Inserts a new Module into the ModuleList at the given index.
        
        Args:
            index : The index to insert the new Module at.
            module : The Module to insert into the ModuleList.
        '''
        self.modules_list.insert(index, module)  
        
    def _walk_module_tree(self: ModuleList) -> list[tuple[str, Module]]:
        '''Walks the full Module tree of the ModuleDict and returns all items.
        
        This method recurses each module contained within the ModuleDict to
        return a list containing not just the modules directly owned by the
        ModuleDict, but also all of their submodules.
        
        Returns:
            list[tuple[str, Module]] : A list of every Module owned by the
                ModuleDict (and all of their submodules), as 
                tuple[str, Module] where the Module is a reference to the 
                Module itself, and the string is the Module's full name.
                
        '''
        result = [('', self)]
        for i, module in enumerate(self.modules_list):
            if isinstance(module, Module):
                for subname, submodule in module._walk_module_tree():
                    full_name = f'{i}.{subname}' if subname else str(i)
                    result.append((full_name, submodule))
        return result
        
    def __getitem__(self: ModuleList, index: int) -> Module:
        '''Gets a reference to the Module at the given index and returns it.
        
        Args:
            index : The index of the Module to get a reference to.
            
        Returns:
            Module : The Module at the given index.
            
        Raises:
            IndexError : If the given index is not present in the ModuleList.
        '''
        return self.modules_list[index]

    def __setitem__(self: ModuleList, index: int, module: Module) -> None:
        '''Sets the given index in the ModuleList to the provided Module.
        
        Args:
            index : The index to set to the new Module.
            module : The Module to set the given index to.
            
        Raises:
            IndexError : If the given index is not present in the ModuleList.
        '''
        self.modules_list[index] = module

    def __delitem__(self: ModuleList, index: int) -> None:
        '''Deletes the item at the given index from the ModuleList.
        
        Args:
            index : The index of the Module to delete.
            
        Raises:
            IndexError : If the given index is not present in the ModuleList.
        '''
        del self.modules_list[index]

    def __len__(self: ModuleList) -> int:
        '''Returns the length of the ModuleList as an integer.
        
        Returns:
            int : The length of the ModuleList.
            
        Raises:
            IndexError : If the given index is not present in the ModuleList.
        '''
        return len(self.modules_list)

    def __iter__(self: ModuleList) -> Iterator[Module]:
        '''Iterates the ModuleList.
        
        Returns:
            Iterator[Module] : Returns an Iterator of the Modules contained
                within the ModuleList.
        '''
        return iter(self.modules_list)

    def __contains__(self: ModuleList, module: Module) -> bool:
        '''Checks whether a given Module is contained within the ModuleList.
        
        Args:
            module : The Module to check for in the ModuleList.
        
        Returns:
            bool : True if the given Module is present in the ModuleList,
                otherwise False.
        '''
        return module in self.modules_list

class Sequential(ModuleList):
    def __init__(
        self:     Sequential, 
        *modules: Module | Iterable[Module] | OrderedDict[str, Module]
    ) -> None:
        '''A special version of a ModuleList which can be called.
        
        When a Sequential is called, it will iterate through its internal
        Module list, calling each Module's forward() function sequentially,
        passing the output of one Module into the next, then returning the 
        final result.
        
        Args:
            modules : The modules to add to the Sequential instance.
        '''
        if len(modules) == 1:
            if isinstance(modules[0], Module): modules = [modules]
            if isinstance(modules[0], OrderedDict):
                modules = list(modules[0].values())
            if isinstance(modules[0], list | tuple):
                modules = list(modules[0])
        else: modules = list(modules)
        super().__init__(modules)
    
    def pop(self: Sequential, index: int) -> Module:
        '''Pops the Module at the index from the Sequential and returns it.
        
        Args:
            index : The index of the Module to pop from the Sequential.
            
        Returns:
            Module : The module popped from the Sequential.
            
        Raises:
            IndexError : If the given index is not present in the Sequential's
                internal Module list.
        '''
        return self.modules_list.pop(index)
    
    def forward(self: Sequential, *inputs: Tensor | Iterable[Tensor]) -> Any:
        '''Calls each Module in the Sequential on the inputs.
        
        Each Module in the Sequential will be called in order, passing the 
        output from one into the next as input, then finally returning the 
        result after all Modules in the Sequential have been called.
        
        Args:
            inputs : The Tensors(s) or Iterable[Tensor] to use as initial input
                for the first Module in the Sequential.
            
        Returns:
            Any : The result after every Module in the Sequential has been 
                called.
        '''
        if not self.modules_list:
            return inputs[0] if len(inputs) == 1 else inputs
        out = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            out = module(out)
        return out
    

