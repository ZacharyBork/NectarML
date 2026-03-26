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

def conv_transpose1d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    B: int, C_in: int, L_in: int,
    C_out: int, K: int,
    stride: int, padding: int, dilation: int,
    output_padding: int,
    groups: int
) -> tuple[np.ndarray, np.ndarray]:
    L_full = (L_in - 1) * stride + dilation * (K - 1) + 1
    L_out  = L_full - 2 * padding + output_padding
    output = np.zeros((B, C_out, L_full), dtype=input.dtype)

    group_in  = C_in  // groups
    group_out = C_out // groups

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_input  = input[:, in_start:in_start + group_in, :]
        g_weight = weight[in_start:in_start + group_in, :, :]

        for n in range(L_in):
            for k in range(K):
                out_pos = n * stride + k * dilation
                if 0 <= out_pos < L_full:
                    output[
                        :, out_start:out_start + group_out, out_pos
                    ] += g_input[:, :, n] @ g_weight[:, :, k]

    output = output[:, :, padding : padding + L_out]
    if bias is not None: output += bias[np.newaxis, :, np.newaxis]
    return output

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
    C_out, C_in, K = weight.shape
    _, _, L_out = grad_output.shape

    result = conv_transpose1d(
        grad_output, weight, None,
        B, C_out, L_out,
        C_in_padded, K,
        stride, padding, dilation,
        output_padding=0, groups=groups)
    return result

def conv_transpose1d_backward_input(
    grad_output: np.ndarray,
    weight: np.ndarray,
    B: int, C_in: int, L_in: int,
    C_out: int, K: int,
    L_out: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int
) -> np.ndarray:
    result, _ = conv1d(
        grad_output, weight, None,
        B, C_out, L_in,
        C_in, K,
        stride, padding, dilation, groups)
    return result

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
  
def conv_transpose1d_backward_weight(
    grad_output: np.ndarray,
    input_padded: np.ndarray,
    B: int, C_in: int, L_in: int,
    C_out: int, K: int,
    L_out: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int
) -> np.ndarray:
    if padding > 0:
        grad_output_full = np.pad(
            grad_output, ((0,0),(0,0),(padding,padding)))
    else:
        grad_output_full = grad_output

    group_in  = C_in  // groups
    group_out = C_out // groups

    grad_weight = np.zeros((C_in, C_out, K), dtype=grad_output.dtype)

    for g in range(groups):
        in_start  = g * group_in
        out_start = g * group_out
        g_input = input_padded[:, in_start:in_start + group_in, :]
        g_grad  = grad_output_full[:, out_start:out_start + group_out, :]

        for k in range(K):
            patches = g_grad[
                :, :, k*dilation : k*dilation + L_in*stride : stride]
            grad_weight[
                in_start:in_start + group_in,
                out_start:out_start + group_out, k
            ] = np.einsum('bin,bon->io', g_input, patches)

    return grad_weight

