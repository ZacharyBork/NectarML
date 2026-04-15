from . import data

from .save       import save, load, save_checkpoint, load_checkpoint
from .inspection import is_inf, is_finite, is_nan, has_inf, has_nan
from .benchmark  import (
    benchmark_time, benchmark_memory, benchmark_host_memory,
    benchmark_device_memory)
