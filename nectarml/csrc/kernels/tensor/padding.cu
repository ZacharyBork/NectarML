#include "kernels/common.h"
#include "include/common/functions.h"

enum class PadMode {
    Constant,
    Replicate,
    Reflect,
    Circular
};

template<typename T, PadMode mode>
__global__ void pad_kernel(
    T*    input, 
    T*    output,
    int   B, 
    int   C,
    int*  in_sizes,
    int*  out_sizes,
    int*  pad_before,
    int   n_dims,
    float constant_value,
    int   total_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;
    
    int spatial_total = total_out / (B * C);
    int bc = idx / spatial_total;
    int spatial_flat = idx % spatial_total;
    
    int out_coords[MAX_DIMS];
    int tmp = spatial_flat;
    for (int d = n_dims - 1; d >= 0; d--) {
        out_coords[d] = tmp % out_sizes[d];
        tmp /= out_sizes[d];
    }

    int in_coords[MAX_DIMS];
    bool is_padding = false;
    
    for (int d = 0; d < n_dims; d++) {
        int in_coord = out_coords[d] - pad_before[d];
        int size = in_sizes[d];

        if constexpr (mode == PadMode::Constant) {
            if (in_coord < 0 || in_coord >= size) {
                is_padding = true;
                break;
            }
            in_coords[d] = in_coord;
        }
        else if constexpr (mode == PadMode::Replicate) {
            in_coords[d] = max(0, min(in_coord, size - 1));
        }
        else if constexpr (mode == PadMode::Reflect) {
            if (in_coord < 0) in_coord = -in_coord;
            if (in_coord >= size) in_coord = 2 * (size - 1) - in_coord;
            in_coords[d] = in_coord;
        }
        else if constexpr (mode == PadMode::Circular) {
            in_coord = in_coord % size;
            if (in_coord < 0) in_coord += size;
            in_coords[d] = in_coord;
        }
    }

    if constexpr (mode == PadMode::Constant) {
        if (is_padding) {
            output[idx] = static_cast<T>(constant_value);
            return;
        }
    }

    int in_spatial_flat = 0;
    int stride = 1;
    for (int d = n_dims - 1; d >= 0; d--) {
        in_spatial_flat += in_coords[d] * stride;
        stride *= in_sizes[d];
    }

    int in_spatial_total = stride;
    output[idx] = input[bc * in_spatial_total + in_spatial_flat];
}

template<typename T>
void launch_pad(
    T*    input, 
    T*    output,
    int   B, 
    int   C,
    int*  in_sizes,
    int*  out_sizes,
    int*  pad_before,
    int   n_dims,
    float constant_value,
    int   total_out,
    const std::string& mode
) {
    int threads = BLOCK_SIZE_1D;
    int blocks = (total_out + threads - 1) / threads;

    if (mode == "constant")
        pad_kernel<T, PadMode::Constant><<<blocks, threads>>>(
            input, output, B, C, in_sizes, out_sizes, 
            pad_before, n_dims, constant_value, total_out);
    else if (mode == "replicate")
        pad_kernel<T, PadMode::Replicate><<<blocks, threads>>>(
            input, output, B, C, in_sizes, out_sizes,
            pad_before, n_dims, constant_value, total_out);
    else if (mode == "reflect")
        pad_kernel<T, PadMode::Reflect><<<blocks, threads>>>(
            input, output, B, C, in_sizes, out_sizes,
            pad_before, n_dims, constant_value, total_out);
    else if (mode == "circular")
        pad_kernel<T, PadMode::Circular><<<blocks, threads>>>(
            input, output, B, C, in_sizes, out_sizes,
            pad_before, n_dims, constant_value, total_out);
}

template void launch_pad<float>(
    float*, float*, int, int, int*, int*, int*, 
    int, float, int, const std::string&
);
template void launch_pad<half>(
    half*, half*, int, int, int*, int*, int*, 
    int, float, int, const std::string&
);
template void launch_pad<uint8_t>(
    uint8_t*, uint8_t*, int, int, int*, int*, int*, 
    int, float, int, const std::string&
);
template void launch_pad<int32_t>(
    int32_t*, int32_t*, int, int, int*, int*, int*, 
    int, float, int, const std::string&
);

template<PadMode mode>
__global__ void pad_backward_kernel(
    float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int  B, 
    int  C,
    int* in_sizes,
    int* out_sizes,
    int* pad_before,
    int  n_dims,
    int  total_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int spatial_total = total_out / (B * C);
    int bc            = idx / spatial_total;
    int spatial_flat  = idx % spatial_total;

    int out_coords[MAX_DIMS];
    int tmp = spatial_flat;
    for (int d = n_dims - 1; d >= 0; d--) {
        out_coords[d] = tmp % out_sizes[d];
        tmp /= out_sizes[d];
    }

    int in_coords[MAX_DIMS];

    for (int d = 0; d < n_dims; d++) {
        int in_coord = out_coords[d] - pad_before[d];
        int size     = in_sizes[d];

        if constexpr (mode == PadMode::Constant) {
            if (in_coord < 0 || in_coord >= size) return;
            in_coords[d] = in_coord;
        }
        else if constexpr (mode == PadMode::Replicate) {
            in_coords[d] = max(0, min(in_coord, size - 1));
        }
        else if constexpr (mode == PadMode::Reflect) {
            if (in_coord < 0)    in_coord = -in_coord;
            if (in_coord >= size) in_coord = 2 * (size - 1) - in_coord;
            in_coords[d] = in_coord;
        }
        else if constexpr (mode == PadMode::Circular) {
            in_coord = in_coord % size;
            if (in_coord < 0) in_coord += size;
            in_coords[d] = in_coord;
        }
    }

    int in_spatial_flat = 0;
    int stride = 1;
    for (int d = n_dims - 1; d >= 0; d--) {
        in_spatial_flat += in_coords[d] * stride;
        stride *= in_sizes[d];
    }

    int in_spatial_total = stride;
    int in_idx = bc * in_spatial_total + in_spatial_flat;

    atomic_add(&grad_in[in_idx], static_cast<float>(grad_out[idx]));
}

void launch_pad_backward(
    float* grad_out,
    float* grad_in,
    int    B, 
    int    C,
    int*   in_sizes,
    int*   out_sizes,
    int*   pad_before,
    int    n_dims,
    int    total_out,
    const std::string& mode
) {
    int threads = BLOCK_SIZE_1D;
    int blocks  = (total_out + threads - 1) / threads;

    if (mode == "constant")
        pad_backward_kernel<PadMode::Constant><<<blocks, threads>>>(
            grad_out, grad_in, B, C, in_sizes, out_sizes,
            pad_before, n_dims, total_out
        );
    else if (mode == "replicate")
        pad_backward_kernel<PadMode::Replicate><<<blocks, threads>>>(
            grad_out, grad_in, B, C, in_sizes, out_sizes,
            pad_before, n_dims, total_out
        );
    else if (mode == "reflect")
        pad_backward_kernel<PadMode::Reflect><<<blocks, threads>>>(
            grad_out, grad_in, B, C, in_sizes, out_sizes,
            pad_before, n_dims, total_out
        );
    else if (mode == "circular")
        pad_backward_kernel<PadMode::Circular><<<blocks, threads>>>(
            grad_out, grad_in, B, C, in_sizes, out_sizes,
            pad_before, n_dims, total_out
        );
}



