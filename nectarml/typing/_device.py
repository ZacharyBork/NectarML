from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml.typing import DeviceLikeType
    
import builtins
from   dataclasses import dataclass
from   typing      import Self, Literal, overload

### DEVICE CLASS ###

_DEVICE_CACHE: dict[tuple, device] = {}

@dataclass
class device:
    type:      Literal['cpu', 'cuda']
    device_id: builtins.int | None = None

    def __new__(
        cls:       type[Self], 
        type:      DeviceLikeType | builtins.str, 
        device_id: builtins.int   | None = None
    ) -> None:
        if isinstance(type, device):
            type      = type.type
            device_id = device_id if device_id is not None else type.device_id
        if type == 'cuda' and device_id is None: device_id = 0
        if type == 'cpu': device_id = None
        
        key = (type, device_id)
        if key in _DEVICE_CACHE:
            return _DEVICE_CACHE[key]
        
        obj = object.__new__(cls)
        _DEVICE_CACHE[key] = obj
        return obj

    @overload
    def __init__(self, type: DeviceLikeType) -> None: ...
    @overload
    def __init__(
        self:      device, 
        type:      builtins.str, 
        device_id: builtins.int | None = None
    ) -> None: ...
    def __init__(
        self:      device, 
        type:      builtins.str,
        device_id: builtins.int | None = None
    ) -> None:
        if hasattr(self, 'type'): return
        
        if isinstance(type, device):
            device_id = device_id if device_id is not None else type.device_id
            type = type.type
            
        assert type in ['cpu', 'cuda'], f'Device type not valid: {type}'
        if device_id is not None:
            assert isinstance(device_id, builtins.int) and device_id >= 0, \
                'device_id must be an integer value >= 0.'
                
        super().__setattr__('type', type)
        super().__setattr__('device_id', device_id)
    
    def __eq__(self: device, other: DeviceLikeType) -> builtins.bool:
        if isinstance(other, str):
            return self.type == other
        if isinstance(other, device):
            return self.type      == other.type \
               and self.device_id == other.device_id
        return NotImplemented

    def __hash__(self: device) -> builtins.int:
        return hash((self.type, self.device_id))

    def __str__(self: device) -> builtins.str:
        if self.device_id is not None:
            return f'{self.type}:{self.device_id}'
        return self.type

    def __repr__(self: device) -> builtins.str:
        return (
            f'nectarml.device('
            f'type="{self.type}", '
            f'device_id={self.device_id})'
        )


