#include "common.h"

namespace py = pybind11;

py::tuple get_cuda_meminfo();

void cuda_synchronize();
void free_cuda(uintptr_t ptr);
void memcpy_to_cuda(uintptr_t dst, uintptr_t src, size_t size_bytes);
uintptr_t alloc_cuda_empty_raw(size_t size_bytes);
uintptr_t alloc_cuda_full(size_t n_elements, DType dtype, double fill_value);
uintptr_t alloc_cuda_random(size_t n_elements, DType dtype, unsigned long long seed, float min_value, float max_value);
uintptr_t alloc_cuda_empty(size_t n_elements, DType dtype);

void register_memory(py::module_& m) {
    m.def("cuda_synchronize", &cuda_synchronize, 
        "Blocks CUDA tasks until the device has finished the current task.");

    m.def("free_cuda", &free_cuda, 
        py::arg("ptr"),
        "Frees GPU memory of object at at given pointer.");

    m.def("memcpy_to_cuda", &memcpy_to_cuda, 
        py::arg("dst"), py::arg("host_ptr"), py::arg("size_bytes"),
        "Directly copies memory from a given address on host to a given address on device.");

    m.def("alloc_cuda_empty_raw", &alloc_cuda_empty_raw, 
        py::arg("size_bytes"),
        "Directly allocates a specified number of bytes in CUDA memory, and returns pointer to allocated memory.");

    m.def("get_cuda_meminfo", &get_cuda_meminfo, 
        "Returns tuple of VRAM statistics: (total, free, used)");

    m.def("alloc_cuda_full", &alloc_cuda_full, 
        py::arg("n_elements"),
        py::arg("dtype"),
        py::arg("fill_value"),
        "Allocates CUDA memory for tensor data of n_elements, and fills with fill_value.");

    m.def("alloc_cuda_random", &alloc_cuda_random, 
        py::arg("n_elements"),
        py::arg("dtype"),
        py::arg("seed"),
        py::arg("min_value"),
        py::arg("max_value"),
        "Allocates memory for CUDA tensor of n_elements and fills with random values from a uniform distribution.");

    m.def("alloc_cuda_empty", &alloc_cuda_empty, 
        py::arg("n_elements"),
        py::arg("dtype"),
        "Allocates CUDA memory for tensor data of n_elements.");
}
