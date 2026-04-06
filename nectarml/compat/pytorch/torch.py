import nectarml
import torch

from nectarml.compat.pytorch.mappings import \
    DTYPES_TORCH2NECTAR, DTYPES_NECTAR2TORCH

def convert_dtype(
    original: torch.dtype | nectarml.DTypeLike
) -> torch.dtype | nectarml.DTypeLike:
    DTYPES = DTYPES_TORCH2NECTAR if isinstance(original, torch.dtype) \
        else DTYPES_NECTAR2TORCH
    try: new_dtype = DTYPES[original]
    except: raise ValueError(f'Unrecognized dtype: {original}')
    return new_dtype

def tensor_torch2nectar(input: torch.Tensor) -> nectarml.Tensor:
    data = input.cpu().numpy()
    dtype = convert_dtype(input.dtype)
    device = 'cuda' if input.device.type == 'cuda' else 'cpu'
    return nectarml.Tensor(
        data, dtype=dtype, device=device, requires_grad=input.requires_grad)
    
def tensor_nectar2torch(input: nectarml.Tensor) -> torch.Tensor:
    new = torch.Tensor(input.cpu().numpy())
    new = new.to(input.device, dtype=convert_dtype(input.dtype))
    new.requires_grad_(input.requires_grad)
    return new

def module_torch2nectar(input: torch.Tensor) -> nectarml.Tensor:
    pass
    
def module_nectar2torch(input: nectarml.Tensor) -> torch.Tensor:
    pass
    
