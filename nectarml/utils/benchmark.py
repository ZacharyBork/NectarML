import os
import time
import psutil
from contextlib import contextmanager
from collections.abc import Iterator

@contextmanager
def benchmark_time(
    operation_name: str | None = None,
    new_line: bool = False
) -> Iterator[None]:
    '''Context to track execution time of code blocks.
    
    Tracks execution time of code block in context and prints result to console
    with optional tag for operation name.
    
    Args:
        operation_name : (Optional) A tag assigned to the operation which will
            be prepended in parentheses to the printed statement if present.
        new_line : If True, a blank line will be printed after the time line
            allowing you to automaticall break up printed statements when
            executing context in loops.
            
    Returns:
        Iterator[None] : An iterator of NoneType values. Default return for
            contextmanager.
    '''
    start_time = None
    try: start_time = time.perf_counter()
    finally:
        yield operation_name, new_line
        if start_time is not None:
            op_tag = '' if operation_name is None else f'({operation_name}) '
            cr = '\n' if new_line else ''
            total = time.perf_counter() - start_time
            print(f'{op_tag}Execution time: {total:.4f} seconds{cr}')
        else: print('Time benchmarking failed.')
        
def benchmark_system_memory(
    operation_name: str | None = None,
    new_line: bool = False
) -> None:
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024
    cr = '\n' if new_line else ''
    print(f'[{operation_name}] RAM: {mb:.1f} MB{cr}')
