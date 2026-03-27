import numpy as np

from nectarml.tensor import Tensor
from nectarml.typing import int32
from nectarml.cpu import pooling as cpu_pool
from nectarml.cuda import pooling as cuda_pool

### AVERAGE POOL ###

def avg_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True
) -> Tensor:
    K = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    S = (K if stride is None else stride) if isinstance(stride, int) \
        else (stride[0] if stride is not None else K)
    P = padding if isinstance(padding, int) else padding[0]

    B, C, L = input.shape
    if ceil_mode: L_out = int(np.ceil((L + 2*P - K) / S + 1))
    else: L_out = int(np.floor((L + 2*P - K) / S + 1))

    input_requires_grad = input.requires_grad

    if input.device == 'cuda':
        out_data = cuda_pool.avg_pool1d_forward(
            input, B, C, L, L_out, K, S, P, count_include_pad)
    else:
        out_data = cpu_pool.avg_pool1d_forward(
            input, B, C, L, L_out, K, S, P, count_include_pad)
    out = Tensor(out_data, (B, C, L_out), input.dtype, input.device,
        requires_grad=input_requires_grad, _children=(input,))

    def _backward() -> None:
        if input_requires_grad:
            out_grad = out.grad.contiguous()
            if input.device == 'cuda':
                grad_ptr = cuda_pool.avg_pool1d_backward(
                    out_grad, B, C, L, L_out, K, S, P, count_include_pad)
                input.grad += Tensor(
                    grad_ptr, input.shape, input.dtype, 'cuda')
            else:
                grad_data = cpu_pool.avg_pool1d_backward(
                    out_grad, B, C, L, L_out, K, S, P, count_include_pad)
                input.grad += Tensor(
                    grad_data, input.shape, input.dtype, 'cpu')
    
    out._backward = _backward
    return out

def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | float | None = None
) -> Tensor:
    KH, KW = (kernel_size, kernel_size) \
        if isinstance(kernel_size, int) else kernel_size
    if stride is None: SH, SW = KH, KW
    elif isinstance(stride, int): SH, SW = stride, stride
    else: SH, SW = stride
    PH, PW = (padding, padding) if isinstance(padding, int) else padding

    B, C, H, W = input.shape
    if ceil_mode:
        H_out = int(np.ceil((H + 2*PH - KH) / SH + 1))
        W_out = int(np.ceil((W + 2*PW - KW) / SW + 1))
    else:
        H_out = int(np.floor((H + 2*PH - KH) / SH + 1))
        W_out = int(np.floor((W + 2*PW - KW) / SW + 1))

    input_requires_grad = input.requires_grad

    if input.device == 'cuda':
        out_data = cuda_pool.avg_pool2d_forward(
            input, B, C, H, W, H_out, W_out,
            KH, KW, SH, SW, PH, PW, count_include_pad)
    else:
        out_data = cpu_pool.avg_pool2d_forward(
            input, B, C, H, W, H_out, W_out,
            KH, KW, SH, SW, PH, PW, count_include_pad, divisor_override)
    out = Tensor(out_data, (B, C, H_out, W_out), input.dtype, input.device,
        requires_grad=input_requires_grad, _children=(input,))

    def _backward() -> None:
        if input_requires_grad:
            out_grad = out.grad.contiguous()
            if input.device == 'cuda':
                grad_ptr = cuda_pool.avg_pool2d_backward(
                    out_grad, B, C, H, W, H_out, W_out,
                    KH, KW, SH, SW, PH, PW, count_include_pad)
                input.grad += Tensor(
                    grad_ptr, input.shape, input.dtype, 'cuda')
            else:
                grad_data = cpu_pool.avg_pool2d_backward(
                    out_grad, B, C, H, W, H_out, W_out,
                    KH, KW, SH, SW, PH, PW, 
                    count_include_pad, divisor_override)
                input.grad += Tensor(
                    grad_data, input.shape, input.dtype, 'cpu')
        
    out._backward = _backward
    return out


def avg_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | float | None = None
) -> Tensor:
    KD, KH, KW = (kernel_size,)*3 \
        if isinstance(kernel_size, int) else kernel_size
    if stride is None: SD, SH, SW = KD, KH, KW
    elif isinstance(stride, int): SD, SH, SW = stride, stride, stride
    else: SD, SH, SW = stride
    PD, PH, PW = (padding,)*3 if isinstance(padding, int) else padding

    B, C, D, H, W = input.shape
    if ceil_mode:
        D_out = int(np.ceil((D + 2*PD - KD) / SD + 1))
        H_out = int(np.ceil((H + 2*PH - KH) / SH + 1))
        W_out = int(np.ceil((W + 2*PW - KW) / SW + 1))
    else:
        D_out = int(np.floor((D + 2*PD - KD) / SD + 1))
        H_out = int(np.floor((H + 2*PH - KH) / SH + 1))
        W_out = int(np.floor((W + 2*PW - KW) / SW + 1))

    input_requires_grad = input.requires_grad

    if input.device == 'cuda':
        out_data = cuda_pool.avg_pool3d_forward(
            input, B, C, D, H, W, D_out, H_out, W_out,
            KD, KH, KW, SD, SH, SW, PD, PH, PW, count_include_pad)
    else:
        out_data = cpu_pool.avg_pool3d_forward(
            input, B, C, D, H, W, D_out, H_out, W_out,
            KD, KH, KW, SD, SH, SW, PD, PH, PW, 
            count_include_pad, divisor_override)
    out = Tensor(out_data, (B, C, D_out, H_out, W_out), input.dtype, 
        input.device, requires_grad=input_requires_grad, 
        _children=(input,))

    def _backward() -> None:
        if input_requires_grad:
            out_grad = out.grad.contiguous()
            if input.device == 'cuda':
                grad_ptr = cuda_pool.avg_pool3d_backward(
                    out_grad, B, C, D, H, W, D_out, H_out, W_out,
                    KD, KH, KW, SD, SH, SW, PD, PH, PW, count_include_pad)
                input.grad += Tensor(
                    grad_ptr, input.shape, input.dtype, 'cuda')
            else:
                grad_data = cpu_pool.avg_pool3d_backward(
                    out_grad, B, C, D, H, W, D_out, H_out, W_out,
                    KD, KH, KW, SD, SH, SW, PD, PH, PW,
                    count_include_pad, divisor_override)
                input.grad += Tensor(
                    grad_data, input.shape, input.dtype, 'cpu')
                
    out._backward = _backward
    return out

### MAX POOL ###

def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] | None = None,
    padding: int | tuple[int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    K = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    S = (K if stride is None else stride) if isinstance(stride, int) \
        else (stride[0] if stride is not None else K)
    P = padding if isinstance(padding, int) else padding[0]
    D = dilation

    B, C, L = input.shape
    if ceil_mode: L_out = int(np.ceil((L + 2*P - D*(K-1) - 1) / S + 1))
    else: L_out = int(np.floor((L + 2*P - D*(K-1) - 1) / S + 1))

    input_requires_grad = input.requires_grad

    if input.device == 'cuda':
        out_data, indices = cuda_pool.max_pool1d_forward(
            input, B, C, L, L_out, K, S, P, D)
    else:
        out_data, indices = cpu_pool.max_pool1d_forward(
            input, B, C, L, L_out, K, S, P, D)

    out = Tensor(out_data, (B, C, L_out), input.dtype, input.device,
        requires_grad=input_requires_grad, _children=(input,))


    def _backward() -> None:
        if input_requires_grad:
            out_grad = out.grad.contiguous()
            if input.device == 'cuda':
                grad_data = cuda_pool.max_pool1d_backward(
                    out_grad, indices, B, C, L, L_out)
            else:
                grad_data = cpu_pool.max_pool1d_backward(
                    out_grad, indices, B, C, L, L_out)
            input.grad += Tensor(grad_data, input.shape, input.dtype,
                input.grad.device)

    out._backward = _backward
    if return_indices:
        idx_tensor = Tensor(indices, (B, C, L_out), int32, 'cuda')
        return out, idx_tensor
    return out

def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    KH, KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) \
             else kernel_size
    if stride is None: SH, SW = KH, KW
    elif isinstance(stride, int): SH, SW = stride, stride
    else: SH, SW = stride
    PH, PW = (padding, padding) if isinstance(padding, int) else padding
    D = dilation

    B, C, H, W = input.shape
    if ceil_mode:
        H_out = int(np.ceil((H + 2*PH - D*(KH-1) - 1) / SH + 1))
        W_out = int(np.ceil((W + 2*PW - D*(KW-1) - 1) / SW + 1))
    else:
        H_out = int(np.floor((H + 2*PH - D*(KH-1) - 1) / SH + 1))
        W_out = int(np.floor((W + 2*PW - D*(KW-1) - 1) / SW + 1))

    input_requires_grad = input.requires_grad

    if input.device == 'cuda':
        out_data, indices = cuda_pool.max_pool2d_forward(
            input, B, C, H, W, H_out, W_out,
            KH, KW, SH, SW, PH, PW, D)
    else:
        out_data, indices = cpu_pool.max_pool2d_forward(
            input, B, C, H, W, H_out, W_out,
            KH, KW, SH, SW, PH, PW, D)
        
    out = Tensor(out_data, (B, C, H_out, W_out), input.dtype, input.device,
        requires_grad=input_requires_grad, _children=(input,))
        
    def _backward() -> None:
        if input_requires_grad:
            out_grad = out.grad.contiguous()
            if input.device == 'cuda':
                grad_data = cuda_pool.max_pool2d_backward(
                    out_grad, indices, B, C, H, W, H_out, W_out)
            else:
                grad_data = cpu_pool.max_pool2d_backward(
                    out_grad, indices, B, C, H, W, H_out, W_out)
            input.grad += Tensor(grad_data, input.shape, input.dtype, 
                input.grad.device)

    out._backward = _backward
    if return_indices:
        idx_tensor = Tensor(
            indices, (B, C, H_out, W_out), int32, input.device)
        return out, idx_tensor
    return out

def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] | None = None,
    padding: int | tuple[int, int, int] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    KD, KH, KW = (kernel_size,)*3 \
        if isinstance(kernel_size, int) else kernel_size
    if stride is None: SD, SH, SW = KD, KH, KW
    elif isinstance(stride, int): SD, SH, SW = stride, stride, stride
    else: SD, SH, SW = stride
    PD, PH, PW = (padding,)*3 if isinstance(padding, int) else padding
    D = dilation

    B, C, Dp, H, W = input.shape
    if ceil_mode:
        D_out = int(np.ceil((Dp + 2*PD - D*(KD-1) - 1) / SD + 1))
        H_out = int(np.ceil((H  + 2*PH - D*(KH-1) - 1) / SH + 1))
        W_out = int(np.ceil((W  + 2*PW - D*(KW-1) - 1) / SW + 1))
    else:
        D_out = int(np.floor((Dp + 2*PD - D*(KD-1) - 1) / SD + 1))
        H_out = int(np.floor((H  + 2*PH - D*(KH-1) - 1) / SH + 1))
        W_out = int(np.floor((W  + 2*PW - D*(KW-1) - 1) / SW + 1))

    input_requires_grad = input.requires_grad

    if input.device == 'cuda':
        out_data, indices = cuda_pool.max_pool3d_forward(
            input, B, C, Dp, H, W, D_out, H_out, W_out,
            KD, KH, KW, SD, SH, SW, PD, PH, PW, D)
    else:
        out_data, indices = cpu_pool.max_pool3d_forward(
            input, B, C, Dp, H, W, D_out, H_out, W_out,
            KD, KH, KW, SD, SH, SW, PD, PH, PW, D)
    out = Tensor(out_data, (B, C, D_out, H_out, W_out), input.dtype,
        input.device, requires_grad=input_requires_grad, _children=(input,))
        
    def _backward() -> None:
        if input_requires_grad:
            out_grad = out.grad.contiguous()
            if input.device == 'cuda':
                grad_data = cuda_pool.max_pool3d_backward(
                    out_grad, indices, B, C, Dp, H, W, D_out, H_out, W_out)
            else:
                grad_data = cpu_pool.max_pool3d_backward(
                    out_grad, indices, B, C, Dp, H, W, D_out, H_out, W_out)

            input.grad += Tensor(grad_data, input.shape, input.dtype, 
                input.grad.device, input_requires_grad, _children=(input,))

    out._backward = _backward
    if return_indices:
        idx_tensor = Tensor(
            indices, (B, C, D_out, H_out, W_out), int32, input.device)
        return out, idx_tensor
    return out

