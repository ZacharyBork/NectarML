import numpy as np

import _nectarml

DTYPE_MAP = {
    np.float32: _nectarml.DType.Float32,
    np.float16: _nectarml.DType.Float16,
    np.int32:   _nectarml.DType.Int32,
    np.uint8:   _nectarml.DType.UInt8,
}

