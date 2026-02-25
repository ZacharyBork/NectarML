import nectarml
import torch

DTYPES_TORCH2NECTAR = {
    torch.float: nectarml.float,
    torch.float16: nectarml.float16,
    torch.float32: nectarml.float32,
    torch.half: nectarml.half,
    torch.double: nectarml.double,
    torch.int: nectarml.int,
    torch.int8: nectarml.int8,
    torch.int16: nectarml.int16,
    torch.int32: nectarml.int32,
    torch.int64: nectarml.int64,
    torch.short: nectarml.short,
    torch.long: nectarml.long,
    torch.uint8: nectarml.uint8,
    torch.uint16: nectarml.uint16,
    torch.uint32: nectarml.uint32,
    torch.uint64: nectarml.uint64
}

DTYPES_NECTAR2TORCH = {
    nectarml.float: torch.float,
    nectarml.float16: torch.float16,
    nectarml.float32: torch.float32,
    nectarml.half: torch.half,
    nectarml.double: torch.double,
    nectarml.int: torch.int,
    nectarml.int8: torch.int8,
    nectarml.int16: torch.int16,
    nectarml.int32: torch.int32,
    nectarml.int64: torch.int64,
    nectarml.short: torch.short,
    nectarml.long: torch.long,
    nectarml.uint8: torch.uint8,
    nectarml.uint16: torch.uint16,
    nectarml.uint32: torch.uint32,
    nectarml.uint64: torch.uint64
}

def convert_dtype(
    original: torch.dtype | nectarml.DTypeLike
) -> torch.dtype | nectarml.DTypeLike:
    DTYPES = DTYPES_TORCH2NECTAR if isinstance(original, torch.dtype) \
        else DTYPES_NECTAR2TORCH
    try: new_dtype = DTYPES[original]
    except: raise ValueError(f'Unrecognized dtype: {original}')
    return new_dtype

def tensor_torch2nectar(input: torch.Tensor) -> nectarml.Tensor:
    data = input.numpy()
    dtype = convert_dtype(input.dtype)
    device = 'cuda' if input.device.type == 'cuda' else 'cpu'
    return nectarml.Tensor(
        data, dtype=dtype, device=device, requires_grad=input.requires_grad)
    
def tensor_nectar2torch(input: nectarml.Tensor) -> torch.Tensor:
    new = torch.Tensor(input.data)
    new = new.to(input.device, dtype=convert_dtype(input.dtype))
    new.requires_grad_(input.requires_grad)
    return new

def module_torch2nectar(input: torch.Tensor) -> nectarml.Tensor:
    pass
    
def module_nectar2torch(input: nectarml.Tensor) -> torch.Tensor:
    pass
    
