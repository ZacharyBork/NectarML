import torch
import nectarml

DTYPES_TORCH2NECTAR = {
    torch.float:   nectarml.float,
    torch.float16: nectarml.float16,
    torch.float32: nectarml.float32,
    torch.half:    nectarml.half,
    torch.double:  nectarml.double,
    torch.int:     nectarml.int,
    torch.int8:    nectarml.int8,
    torch.int16:   nectarml.int16,
    torch.int32:   nectarml.int32,
    torch.int64:   nectarml.int64,
    torch.short:   nectarml.short,
    torch.long:    nectarml.long,
    torch.uint8:   nectarml.uint8,
    torch.uint16:  nectarml.uint16,
    torch.uint32:  nectarml.uint32,
    torch.uint64:  nectarml.uint64
}

DTYPES_NECTAR2TORCH = {
    nectarml.float:   torch.float,
    nectarml.float16: torch.float16,
    nectarml.float32: torch.float32,
    nectarml.half:    torch.half,
    nectarml.double:  torch.double,
    nectarml.int:     torch.int,
    nectarml.int8:    torch.int8,
    nectarml.int16:   torch.int16,
    nectarml.int32:   torch.int32,
    nectarml.int64:   torch.int64,
    nectarml.short:   torch.short,
    nectarml.long:    torch.long,
    nectarml.uint8:   torch.uint8,
    nectarml.uint16:  torch.uint16,
    nectarml.uint32:  torch.uint32,
    nectarml.uint64:  torch.uint64
}

