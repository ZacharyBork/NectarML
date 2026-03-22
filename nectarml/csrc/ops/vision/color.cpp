#include "common.h"

namespace py = pybind11;

template<typename T>
void launch_hsv_adjust(
    T* d_in, int B, int C, int H, int W,
    float hue_shift, float saturation, float value
);

namespace nectar {
    
    uintptr_t hsv_adjust(
        uintptr_t in_ptr, std::vector<int> shape, 
        float hue_shift, float saturation, float value,
        DType dtype
    ) {
        size_t memsize = 1;
        for (int i : shape) { memsize *= i; }

        DISPATCH_DTYPE(dtype, T, {
            T* d_out;
            cudaMalloc(&d_out, memsize * sizeof(T));
            cudaMemcpy(d_out, reinterpret_cast<T*>(in_ptr), 
                   memsize * sizeof(T), cudaMemcpyDeviceToDevice);
            launch_hsv_adjust<T>(
                d_out, shape[0], shape[1], shape[2], shape[3],
                hue_shift, saturation, value);
            return reinterpret_cast<uintptr_t>(d_out);
        });
    }

}

// py::array_t<uint8_t> hue_shift(py::array_t<uint8_t> image, float shift) {
//     // Validate input
//     auto buf = image.request();
//     if (buf.ndim != 3) throw std::runtime_error("Expected a 3D array (H, W, C)");
//     if (buf.shape[2] != 3) throw std::runtime_error("Expected 3 channels (RGB)");

//     int height = buf.shape[0];
//     int width  = buf.shape[1];
//     size_t nbytes = height * width * 3 * sizeof(uint8_t);

//     // Allocate VRAM
//     uint8_t* d_image;
//     cudaMalloc(&d_image, nbytes);

//     // Copy to VRAM
//     cudaMemcpy(d_image, buf.ptr, nbytes, cudaMemcpyHostToDevice);

//     // Run kernel
//     launch_hue_shift(d_image, width, height, shift / 360.0f);

//     // Copy result to DRAM
//     auto result = py::array_t<uint8_t>(buf.shape);
//     auto result_buf = result.request();
//     cudaMemcpy(result_buf.ptr, d_image, nbytes, cudaMemcpyDeviceToHost);

//     // Free VRAM
//     cudaFree(d_image);

//     return result;
// }
