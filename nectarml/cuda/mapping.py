import _nectarml
from nectarml import typing

DTYPE_MAP = {
    typing.float:   _nectarml.DType.Float32,
    typing.float32: _nectarml.DType.Float32,
    typing.float16: _nectarml.DType.Float16,
    typing.int32:   _nectarml.DType.Int32,
    typing.uint8:   _nectarml.DType.UInt8,
}

