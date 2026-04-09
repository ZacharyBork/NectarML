from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nectarml import Tensor

import numpy as np

from nectarml import typing

### AVERAGE POOL ###

def avg_pool1d_forward(
    input: Tensor,
    B: int, C: int, L: int, L_out: int,
    K: int, S: int, P: int,
    count_include_pad: bool
) -> np.ndarray:
    data = input.data
    if P > 0:
        data = np.pad(data, ((0,0),(0,0),(P,P)), mode='constant')
    out = np.zeros((B, C, L_out), dtype=input.dtype)
    for n in range(L_out):
        start  = n * S
        window = data[:, :, start:start+K]
        if count_include_pad:
            out[:, :, n] = window.mean(axis=2)
        else:
            l_start = max(0, n * S - P)
            l_end   = min(L, n * S - P + K)
            count   = l_end - l_start
            out[:, :, n] = window.sum(axis=2) / count
    return out

def avg_pool1d_backward(
    out_grad: Tensor,
    B: int, C: int, L: int, L_out: int,
    K: int, S: int, P: int,
    count_include_pad: bool
) -> np.ndarray:
    grad_input = np.zeros((B, C, L + 2*P), dtype=out_grad.dtype)
    for n in range(L_out):
        start = n * S
        if count_include_pad:
            div = K
        else:
            l_start = max(0, n * S - P)
            l_end   = min(L, n * S - P + K)
            div     = l_end - l_start
        grad_input[:, :, start:start+K] += out_grad.data[:, :, n:n+1] / div
    if P > 0:
        grad_input = grad_input[:, :, P:-P]
    return grad_input


def avg_pool2d_forward(
    input: Tensor,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int,
    KH: int, KW: int,
    SH: int, SW: int,
    PH: int, PW: int,
    count_include_pad: bool,
    divisor_override: int | float | None = None
) -> np.ndarray:
    data = input.data
    if PH > 0 or PW > 0:
        data = np.pad(data, ((0,0),(0,0),(PH,PH),(PW,PW)), mode='constant')
    out = np.zeros((B, C, H_out, W_out), dtype=input.dtype)
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * SH
            w_start = j * SW
            window  = data[:, :, h_start:h_start+KH, w_start:w_start+KW]
            if divisor_override is not None:
                out[:, :, i, j] = window.sum(axis=(2,3)) / divisor_override
            elif count_include_pad:
                out[:, :, i, j] = window.mean(axis=(2,3))
            else:
                h_s = max(0, i * SH - PH); h_e = min(H, i * SH - PH + KH)
                w_s = max(0, j * SW - PW); w_e = min(W, j * SW - PW + KW)
                count = (h_e - h_s) * (w_e - w_s)
                out[:, :, i, j] = window.sum(axis=(2,3)) / count
    return out

def avg_pool2d_backward(
    out_grad: Tensor,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int,
    KH: int, KW: int,
    SH: int, SW: int,
    PH: int, PW: int,
    count_include_pad: bool,
    divisor_override: int | float | None = None
) -> np.ndarray:
    grad_input = np.zeros((B, C, H + 2*PH, W + 2*PW), dtype=out_grad.dtype)
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * SH
            w_start = j * SW
            if divisor_override is not None:
                div = divisor_override
            elif count_include_pad:
                div = KH * KW
            else:
                h_s = max(0, i * SH - PH); h_e = min(H, i * SH - PH + KH)
                w_s = max(0, j * SW - PW); w_e = min(W, j * SW - PW + KW)
                div = (h_e - h_s) * (w_e - w_s)
            grad_input[
                :, :, h_start:h_start+KH, w_start:w_start+KW
            ] += out_grad.data[:, :, i:i+1, j:j+1] / div
    if PH > 0 or PW > 0:
        grad_input = grad_input[:, :, PH:-PH, PW:-PW]
    return grad_input


def avg_pool3d_forward(
    input: Tensor,
    B: int, C: int, D: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int,
    KD: int, KH: int, KW: int,
    SD: int, SH: int, SW: int,
    PD: int, PH: int, PW: int,
    count_include_pad: bool,
    divisor_override: int | float | None = None
) -> np.ndarray:
    data = input.data
    if PD > 0 or PH > 0 or PW > 0:
        data = np.pad(
            data, ((0,0),(0,0),(PD,PD),(PH,PH),(PW,PW)), mode='constant')
    out = np.zeros((B, C, D_out, H_out, W_out), dtype=input.dtype)
    for d in range(D_out):
        for i in range(H_out):
            for j in range(W_out):
                d_start = d * SD
                h_start = i * SH
                w_start = j * SW
                window  = data[
                    :, :,
                    d_start:d_start+KD,
                    h_start:h_start+KH,
                    w_start:w_start+KW]
                if divisor_override is not None:
                    out[:, :, d, i, j] = \
                        window.sum(axis=(2,3,4)) / divisor_override
                elif count_include_pad:
                    out[:, :, d, i, j] = window.mean(axis=(2,3,4))
                else:
                    d_s = max(0, d*SD-PD); d_e = min(D, d*SD-PD+KD)
                    h_s = max(0, i*SH-PH); h_e = min(H, i*SH-PH+KH)
                    w_s = max(0, j*SW-PW); w_e = min(W, j*SW-PW+KW)
                    count = (d_e-d_s) * (h_e-h_s) * (w_e-w_s)
                    out[:, :, d, i, j] = window.sum(axis=(2,3,4)) / count
    return out

def avg_pool3d_backward(
    out_grad: Tensor,
    B: int, C: int, D: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int,
    KD: int, KH: int, KW: int,
    SD: int, SH: int, SW: int,
    PD: int, PH: int, PW: int,
    count_include_pad: bool,
    divisor_override: int | float | None = None
) -> np.ndarray:
    grad_input = np.zeros(
        (B, C, D + 2*PD, H + 2*PH, W + 2*PW), dtype=out_grad.dtype)
    for d in range(D_out):
        for i in range(H_out):
            for j in range(W_out):
                d_start = d * SD
                h_start = i * SH
                w_start = j * SW
                if divisor_override is not None:
                    div = divisor_override
                elif count_include_pad:
                    div = KD * KH * KW
                else:
                    d_s = max(0, d*SD-PD); d_e = min(D, d*SD-PD+KD)
                    h_s = max(0, i*SH-PH); h_e = min(H, i*SH-PH+KH)
                    w_s = max(0, j*SW-PW); w_e = min(W, j*SW-PW+KW)
                    div = (d_e-d_s) * (h_e-h_s) * (w_e-w_s)
                grad_input[
                    :, :,
                    d_start:d_start+KD,
                    h_start:h_start+KH,
                    w_start:w_start+KW
                ] += out_grad.data[:, :, d:d+1, i:i+1, j:j+1] / div
    if PD > 0 or PH > 0 or PW > 0:
        grad_input = grad_input[:, :, PD:-PD, PH:-PH, PW:-PW]
    return grad_input


### MAX POOL ###

def max_pool1d_forward(
    input: Tensor,
    B: int, C: int, L: int, L_out: int,
    K: int, S: int, P: int, D: int
) -> tuple[np.ndarray, np.ndarray]:
    data = input.data
    if P > 0:
        data = np.pad(
            data, ((0,0),(0,0),(P,P)), 
            mode='constant', constant_values=-np.inf)
    out     = np.full((B, C, L_out), -np.inf, dtype=input.dtype)
    indices = np.zeros((B, C, L_out), dtype=np.int64)
    for n in range(L_out):
        positions     = np.arange(K) * D + n * S
        valid         = (positions >= 0) & (positions < data.shape[2])
        pos_valid     = positions[valid]
        window        = data[:, :, pos_valid]
        idx_in_window = np.argmax(window, axis=2)
        out[:, :, n]  = window[
            np.arange(B)[:, None], np.arange(C)[None, :], idx_in_window]
        indices[:, :, n] = pos_valid[idx_in_window] - P
    return out, indices

def max_pool1d_backward(
    out_grad: Tensor,
    indices: np.ndarray,
    B: int, C: int, L: int, L_out: int
) -> np.ndarray:
    grad_input = np.zeros((B, C, L), dtype=out_grad.dtype)
    for n in range(L_out):
        idx = indices[:, :, n]
        np.add.at(grad_input,
                  (np.arange(B)[:, None], np.arange(C)[None, :], idx),
                  out_grad.data[:, :, n])
    return grad_input


def max_pool2d_forward(
    input: Tensor,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int,
    KH: int, KW: int,
    SH: int, SW: int,
    PH: int, PW: int, D: int
) -> tuple[np.ndarray, np.ndarray]:
    data = input.data
    if PH > 0 or PW > 0:
        data = np.pad(
            data, ((0,0),(0,0),(PH,PH),(PW,PW)),
            mode='constant', constant_values=-np.inf)
    out     = np.full((B, C, H_out, W_out), -np.inf, dtype=input.dtype)
    indices = np.zeros((B, C, H_out, W_out), dtype=np.int64)
    h_pos   = np.arange(KH) * D
    w_pos   = np.arange(KW) * D
    for i in range(H_out):
        for j in range(W_out):
            h_positions = h_pos + i * SH
            w_positions = w_pos + j * SW
            window = data[
                :, :,
                h_positions[:, None],
                w_positions[None, :]]
            flat  = window.reshape(B, C, KH * KW)
            idx   = np.argmax(flat, axis=2)
            out[:, :, i, j] = flat[
                np.arange(B)[:, None], np.arange(C)[None, :], idx]
            kh_idx = idx // KW
            kw_idx = idx %  KW
            h_orig = h_positions[kh_idx] - PH
            w_orig = w_positions[kw_idx] - PW
            indices[:, :, i, j] = h_orig * W + w_orig
    return out, indices

def max_pool2d_backward(
    out_grad: Tensor,
    indices: np.ndarray,
    B: int, C: int, H: int, W: int,
    H_out: int, W_out: int
) -> np.ndarray:
    grad_input = np.zeros((B, C, H, W), dtype=out_grad.dtype)
    for i in range(H_out):
        for j in range(W_out):
            idx   = indices[:, :, i, j]
            h_idx = idx // W
            w_idx = idx %  W
            np.add.at(grad_input,
                      (np.arange(B)[:, None], np.arange(C)[None, :],
                       h_idx, w_idx),
                      out_grad.data[:, :, i, j])
    return grad_input


def max_pool3d_forward(
    input: Tensor,
    B: int, C: int, Dp: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int,
    KD: int, KH: int, KW: int,
    SD: int, SH: int, SW: int,
    PD: int, PH: int, PW: int, D: int
) -> tuple[np.ndarray, np.ndarray]:
    data = input.data
    if PD > 0 or PH > 0 or PW > 0:
        data = np.pad(
            data, ((0,0),(0,0),(PD,PD),(PH,PH),(PW,PW)),
            mode='constant', constant_values=-np.inf)
    out     = np.full((B, C, D_out, H_out, W_out), -np.inf, dtype=input.dtype)
    indices = np.zeros((B, C, D_out, H_out, W_out), dtype=np.int64)
    d_pos   = np.arange(KD) * D
    h_pos   = np.arange(KH) * D
    w_pos   = np.arange(KW) * D
    for di in range(D_out):
        for i in range(H_out):
            for j in range(W_out):
                d_positions = d_pos + di * SD
                h_positions = h_pos + i  * SH
                w_positions = w_pos + j  * SW
                window = data[
                    :, :,
                    d_positions[:, None, None],
                    h_positions[None, :, None],
                    w_positions[None, None, :]]
                flat = window.reshape(B, C, KD * KH * KW)
                idx  = np.argmax(flat, axis=2)
                out[:, :, di, i, j] = flat[
                    np.arange(B)[:, None], np.arange(C)[None, :], idx]
                kd_idx = idx // (KH * KW)
                kh_idx = (idx % (KH * KW)) // KW
                kw_idx = idx % KW
                d_orig = d_positions[kd_idx] - PD
                h_orig = h_positions[kh_idx] - PH
                w_orig = w_positions[kw_idx] - PW
                indices[:, :, di, i, j] = \
                    d_orig * H * W + h_orig * W + w_orig
    return out, indices

def max_pool3d_backward(
    out_grad: Tensor,
    indices: np.ndarray,
    B: int, C: int, Dp: int, H: int, W: int,
    D_out: int, H_out: int, W_out: int
) -> np.ndarray:
    grad_input = np.zeros((B, C, Dp, H, W), dtype=out_grad.dtype)
    for di in range(D_out):
        for i in range(H_out):
            for j in range(W_out):
                idx   = indices[:, :, di, i, j]
                d_idx = idx // (H * W)
                h_idx = (idx % (H * W)) // W
                w_idx = idx % W
                np.add.at(grad_input,
                          (np.arange(B)[:, None], np.arange(C)[None, :],
                           d_idx, h_idx, w_idx),
                          out_grad.data[:, :, di, i, j])
    return grad_input

