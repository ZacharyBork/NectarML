import os
import time
import psutil
from   typing import Self, Any

from nectarml import cuda

class benchmark_time:
    def __init__(
        self, 
        operation_name:  str | None = None,
        new_line:        bool = False,
        enabled:         bool = True,
        force_cuda_sync: bool = False
    ) -> None:
        '''Context to track execution time of code blocks.
        
        Tracks execution time of code block in context and prints result to 
        console with optional tag for operation name.
        
        Args:
            operation_name : (Optional) A tag assigned to the operation which
                will be prepended in parentheses to the printed statement if 
                present.
            new_line : If True, a blank line will be printed after the time 
                line allowing you to automaticall break up printed statements 
                when executing context in loops.
                
        Returns:
            Iterator[None] : An iterator of NoneType values. Default return for
                contextmanager.
        '''
        self.op_name     = operation_name
        self.new_line    = new_line
        self.enabled     = enabled
        self._start_time = 0
        self._sync = lambda : None if force_cuda_sync \
                else cuda.utils.cuda_synchronize()
    
    def __enter__(self) -> Self:
        self._sync()
        if self.enabled: self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        if not self.enabled: return 
        self._sync()
        
        op_tag = '' if self.op_name is None else f'({self.op_name}) '
        cr     = '\n' if self.new_line else ''
        total  = time.perf_counter() - self._start_time
        print(f'{op_tag}Execution time: {total:.4f} seconds{cr}')

def benchmark_memory(
    operation_name: str | None = None,
    new_line: bool = False
) -> None:
    '''Prints memory statistic (total, free, used) for both host and device.
    
    Args:
        operation_name : A name string for the operation being benchmarked, or
            None. This string will be prepended to the print output if 
            provided.
        new_line : If True, an additional blank line will be printed after the
            memory info to help split up console output for repeated logging
            operations.
    '''
    cr = '\n' if new_line else ''
    op_name = f'Operation: {operation_name}\n' \
           if operation_name is not None else ''
    
    cpu_process = psutil.Process(os.getpid()).memory_info()
    cpu_virtual = psutil.virtual_memory()
    host_mem = [round(i/1024**3, 2) for i in [
        cpu_virtual.total, cpu_virtual.free, cpu_process.rss]]
    cpu_total, cpu_free, cpu_used = host_mem
  
    cuda_process = cuda.memory.get_cuda_meminfo()
    device_mem = [round(i/1024**3, 2) for i in cuda_process]
    cuda_total, cuda_free, cuda_used = device_mem
    
    print(
        f'{op_name}'
        f'Host:\n'
        f'    Total: {cpu_total}\n'
        f'    Used:  {cpu_used}\n'
        f'    Free:  {cpu_free}\n'
        f'Device:\n'
        f'    Total: {cuda_total}\n'
        f'    Used:  {cuda_used}\n'
        f'    Free:  {cuda_free}'
        f'{cr}'
        
    )
        
def benchmark_host_memory(
    operation_name: str | None = None,
    new_line: bool = False
) -> None:
    '''Prints memory statistic (total, free, used) for host (DRAM).
    
    Args:
        operation_name : A name string for the operation being benchmarked, or
            None. This string will be prepended to the print output if 
            provided.
        new_line : If True, an additional blank line will be printed after the
            memory info to help split up console output for repeated logging
            operations.
    '''
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024    
    cr = '\n' if new_line else ''
    op_name = f'[{operation_name}] ' if operation_name is not None else ''
    print(f'{op_name}RAM: {mb:.1f} MB{cr}')

def benchmark_device_memory(
    operation_name: str | None = None,
    new_line: bool = False
) -> None:
    '''Prints memory statistic (total, free, used) for device (VRAM).
    
    Args:
        operation_name : A name string for the operation being benchmarked, or
            None. This string will be prepended to the print output if 
            provided.
        new_line : If True, an additional blank line will be printed after the
            memory info to help split up console output for repeated logging
            operations.
    '''
    cr = '\n' if new_line else ''
    op_name = f'[{operation_name}] ' if operation_name is not None else ''
    print(f'{op_name}{cuda.memory.get_memory_statistics()}{cr}')
