from typing import Literal
from collections.abc import Callable

import numpy as np

'''
Output Size Formula:

L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

Pseudocode:

for each batch item b:
  for each output channel co:
    for each output position n:
      sum = bias[co]
      for each input channel ci:
        for each kernel position k:
          sum += input[b, ci, n*stride + k*dilation] * kernel[co, ci, k]
      output[b, co, n] = sum
'''

def conv1d_backward(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    B, C_in, _  = input_padded.shape
    C_out, _, K = weight.shape
    _, _, L_out = grad_output.shape

    group_in = C_in // groups
    group_out = C_out // groups

    grad_bias = grad_output.sum(axis=(0, 2)) if bias is not None else None
    grad_weight = np.zeros_like(weight)
    grad_input_padded = np.zeros_like(input_padded)

    for g in range(groups):
        in_start = g * group_in
        out_start = g * group_out

        g_input = input_padded[:, in_start:in_start + group_in, :]
        g_grad = grad_output[:, out_start:out_start + group_out, :]
        g_weight = weight[out_start:out_start + group_out, :, :]

        for k in range(K):
            patches = g_input[
                :, :, k*dilation : k*dilation + L_out*stride : stride]
            grad_weight[out_start:out_start+group_out, :, k] = (
                np.einsum('bon,bin->oi', g_grad, patches))

        for n in range(L_out):
            for k in range(K):
                grad_input_padded[
                    :, in_start:in_start+group_in, n*stride + k*dilation] += (
                    g_grad[:, :, n] @ g_weight[:, :, k])

    if padding > 0: grad_input = grad_input_padded[:, :, padding:-padding]
    else: grad_input = grad_input_padded
    return grad_input, grad_weight, grad_bias

def conv1d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> tuple[
        np.ndarray, 
        Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    
    B, C_in, L_in = input.shape
    C_out, _, K   = weight.shape
    
    assert C_in % groups == 0, (
        f'Input channel count [{C_in}] must be evenly divisible by number '
        f'of groups [{groups}].')
    assert C_out % groups == 0, (
        f'Weight channel count [{C_out}] must be evenly divisible by number '
        f'of groups [{groups}].')
    
    if padding == 'valid': padding = 0
    elif padding == 'same': 
        padding = ((L_in - 1) * stride - L_in + dilation * (K-1) + 1) // 2
    if padding > 0: input = np.pad(input, ((0, 0), (0, 0), (padding, padding)))
    
    L_out = (L_in + 2*padding - dilation*(K-1) - 1) // stride + 1
    output = np.zeros((B, C_out, L_out))
    
    group_in = C_in // groups
    group_out = C_out // groups
    
    for g in range(groups):
        in_start = g * group_in
        out_start = g * group_out
        for batch in range(B):
            for ch_out in range(group_out):
                for position in range(L_out):
                    s = bias[out_start + ch_out] if bias is not None else 0.0
                    
                    for ch_in in range(group_in):
                        for k in range(K):
                            value = input[
                                batch, in_start + ch_in, 
                                position*stride + k*dilation]
                            s += value * weight[out_start + ch_out, ch_in, k]
                            
                    output[batch, out_start + ch_out, position] = s
              
    def _backward(
        out_grad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return conv1d_backward(
            out_grad, input, weight, bias, stride, padding, dilation, groups)
                
    return output, _backward
                    
def conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> np.ndarray:
    pass
    
def conv3d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | Literal['valid', 'same'] = 0,
    dilation: int = 1,
    groups: int = 1
) -> np.ndarray:
    pass





