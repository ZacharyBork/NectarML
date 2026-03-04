#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common.h"

namespace py = pybind11;

void free_cuda(uintptr_t ptr);
uintptr_t alloc_cuda_full(size_t n_elements, DType dtype, double fill_value);
uintptr_t alloc_cuda_random(size_t n_elements, DType dtype, unsigned long long seed, float min_value, float max_value);
uintptr_t alloc_cuda_empty(size_t n_elements, DType dtype);

void register_memory(py::module_& m) {
    m.def("free_cuda", &free_cuda, 
        py::arg("ptr"),
        "Frees GPU memory of object at at given pointer.");

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
