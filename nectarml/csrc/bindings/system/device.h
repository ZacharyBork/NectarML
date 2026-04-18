#pragma once

#include "common/dtype.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace nectar {
    uintptr_t cast_tensor(
        uintptr_t device_ptr, 
        size_t n_elements, 
        DType src_dtype,
        DType dst_dtype
    );

    uintptr_t to_cuda(uintptr_t host_ptr, size_t n_elements, DType dtype);
    py::array to_cpu(uintptr_t device_ptr, std::vector<size_t> shape, DType dtype);
    uintptr_t clone(uintptr_t device_ptr, size_t n_elements, DType dtype);

}

void register_device(py::module_& m) {
    m.def("cast_tensor", &nectar::cast_tensor, 
        py::arg("device_ptr"), 
        py::arg("n_elements"), 
        py::arg("src_dtype"),
        py::arg("dst_dtype"),
        "Changes DType of CUDA tensor data.");

    m.def("to_cuda", &nectar::to_cuda, 
        py::arg("host_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Moves tensor data from system memory to GPU memory.");

    m.def("to_cpu", &nectar::to_cpu, 
        py::arg("device_ptr"),
        py::arg("shape"),
        py::arg("dtype"),
        "Moves tensor data from GPU memory to system memory.");
    
    m.def("clone", &nectar::clone, 
        py::arg("device_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Moves tensor data from GPU memory to system memory.");
}

