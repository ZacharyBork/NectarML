#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include <pybind11/numpy.h>

namespace py = pybind11;

void launch_hue_shift(uint8_t* d_image, int width, int height, float shift);

py::array_t<uint8_t> hue_shift(py::array_t<uint8_t> image, float shift) {
    // Validate input
    auto buf = image.request();
    if (buf.ndim != 3) throw std::runtime_error("Expected a 3D array (H, W, C)");
    if (buf.shape[2] != 3) throw std::runtime_error("Expected 3 channels (RGB)");

    int height = buf.shape[0];
    int width  = buf.shape[1];
    size_t nbytes = height * width * 3 * sizeof(uint8_t);

    // Allocate VRAM
    uint8_t* d_image;
    cudaMalloc(&d_image, nbytes);

    // Copy to VRAM
    cudaMemcpy(d_image, buf.ptr, nbytes, cudaMemcpyHostToDevice);

    // Run kernel
    launch_hue_shift(d_image, width, height, shift / 360.0f);

    // Copy result to DRAM
    auto result = py::array_t<uint8_t>(buf.shape);
    auto result_buf = result.request();
    cudaMemcpy(result_buf.ptr, d_image, nbytes, cudaMemcpyDeviceToHost);

    // Free VRAM
    cudaFree(d_image);

    return result;
}
