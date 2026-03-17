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

### 1-Dimensional ###

def conv1d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    B: int, C_in: int, L_out: int,
    C_out: int, K: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    if padding > 0: input = np.pad(input, ((0, 0), (0, 0), (padding, padding)))
    output = np.zeros((B, C_out, L_out), dtype=input.dtype)
    group_in  = C_in  // groups
    group_out = C_out // groups

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        for b in range(B):
            for co in range(group_out):
                for n in range(L_out):
                    s = bias[out_start + co] if bias is not None else 0.0
                    for ci in range(group_in):
                        for k in range(K):
                            s += (
                                input[
                                    b, in_start+ci, n*stride + k*dilation
                                ] * weight[out_start + co, ci, k])
                    output[b, out_start + co, n] = s

    return output, input

def conv1d_backward_input(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    weight: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> np.ndarray:
    B, C_in_padded, _ = input_padded.shape
    C_out, _, K = weight.shape
    _, _, L_out = grad_output.shape

    group_in  = C_in_padded // groups
    group_out = C_out // groups

    grad_input_padded = np.zeros_like(input_padded)

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_grad   = grad_output[:, out_start:out_start + group_out, :]
        g_weight = weight[out_start:out_start + group_out, :, :]

        for n in range(L_out):
            for k in range(K):
                grad_input_padded[
                    :, in_start:in_start + group_in, n*stride + k*dilation
                ] += g_grad[:, :, n] @ g_weight[:, :, k]

    if padding > 0:
        return grad_input_padded[:, :, padding:-padding]
    return grad_input_padded


def conv1d_backward_weight(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    weight: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> np.ndarray:
    B, C_in_padded, _ = input_padded.shape
    C_out, _, K = weight.shape
    _, _, L_out = grad_output.shape

    group_in  = C_in_padded // groups
    group_out = C_out // groups

    grad_weight = np.zeros_like(weight)

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_input  = input_padded[:, in_start:in_start + group_in, :]
        g_grad   = grad_output[:, out_start:out_start + group_out, :]

        for k in range(K):
            patches = g_input[
                :, :, k*dilation : k*dilation + L_out*stride : stride]
            grad_weight[out_start:out_start + group_out, :, k] = (
                np.einsum('bon,bin->oi', g_grad, patches))

    return grad_weight
  
### 2-Dimensional ###
  
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
    
### 3-Dimensional ###
    
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





