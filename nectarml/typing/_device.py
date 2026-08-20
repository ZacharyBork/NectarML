from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml.typing import DeviceLikeType
    
import builtins
from   dataclasses import dataclass
from   typing      import Self, Literal

### DEVICE CLASS ###

_DEVICE_CACHE: dict[tuple, device] = {}

@dataclass
class device:
    type:      Literal['cpu', 'cuda']
    device_id: builtins.int | None = None

    def __new__(
        cls:       type[Self], 
        type:      DeviceLikeType, 
        device_id: builtins.int | None = None
    ) -> Self:
        '''Sets device id for CUDA and checks device cache to avoid duplicates.

        If passed another `device` object for initialization, this method will
        inherit the device id of the given `device`. If passed a string (i.e.
        "cuda", "cpu"), and if `device_id` if not provided, this method will 
        set the device id to None for CPU tensors, and 0 for CUDA tensors. 
        Otherwise it will set the device id of the new `device` to the 
        specified `device_id`.

        This method will also check against the `_DEVICE_CACHE` upon creation
        of a new `device` object to see if an existing `device` object with the
        same specifications already exists. If so, it will instead return a 
        reference to the existing object in the cache to avoid the unnecessary 
        creation of a duplicate `device`.

        Please note that the above means that any time a new `device` is 
        initialized by passing an existing device, and a `device_id` is not
        specified, this method will always return a reference to the device 
        passed for initialization. This allows you to do things like:
        
        ```
        from nectarml.typing import device, DeviceLikeType

        def make_device(
            in_device: DeviceLikeType, 
            device_id: int | None = None
        ) -> device:
            return device(in_device, device_id)

        x = make_device('cuda') # Makes new cuda device and assigns to x
        y = make_device(x)      # Same function now makes y a reference to "x"
        ```
        
        Args:
            type      : The device type for the new `device` object. Can be 
                        another `device` object, or ["cpu", "cuda"]
            device_id : The device id for the new device, or None. Only used 
                        for CUDA devices.
        
        Returns:
            device : The new (or cached) device object.
        '''
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

    def __init__(
        self:      device, 
        type:      DeviceLikeType,
        device_id: builtins.int | None = None
    ) -> None:
        '''Initailizes a new device object.

        Args:
            type      : The device type for the new `device` object. Can be 
                        another `device` object, or ["cpu", "cuda"]
            device_id : The device id for the new device, or None. Only used 
                        for CUDA devices.
        '''
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
        '''Checks if a DeviceLikeType object is the same as a given device.

        Can check device<->device and device<->string. So both of these are
        valid:
        ```
        from nectarml.typing import device
        
        x = device('cuda')
        y = device('cpu')
        print(x == y) # Prints "False"
        ```
        And:
        ```
        from nectarml.typing import device
        
        x = device('cuda')
        print(x == 'cuda') # Prints "True"
        ```
        
        Args:
            other : The DeviceLikeType object to check the device against.
        
        Returns:
            bool : True if the given device is equal to the DeviceLikeType 
                   object being compared agains, otherwise False.
        '''
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


