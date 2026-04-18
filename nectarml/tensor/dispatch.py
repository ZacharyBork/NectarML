import builtins
from typing import Any
from collections.abc import Callable

import numpy as np

from nectarml import cuda, cpu, typing

def _dispach_function_dynamic(
    fn:          Callable[[Any], np.ndarray | builtins.int],
    device:      typing.device,
    *fn_args:    Any,
    **fn_kwargs: dict[builtins.str, Any]
) -> np.ndarray | builtins.int:
    match device.type:
        case 'cpu':  pass
        case 'cuda': pass
        case _:
            raise ValueError(
                f'_dispatch_function_dynamic receieved a device with an '
                f'unrecognized type: {device.type}')





