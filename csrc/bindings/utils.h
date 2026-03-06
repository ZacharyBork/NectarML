#include <pybind11/pybind11.h>
#include "common.h"

namespace py = pybind11;

namespace nectar {
    float compute_tensor_min(uintptr_t device_ptr, size_t n_elements, DType dtype);
    float compute_tensor_max(uintptr_t device_ptr, size_t n_elements, DType dtype);
    std::vector<float> compute_tensor_range(uintptr_t device_ptr, size_t n_elements, DType dtype);
}

void register_utils(py::module_& m) {

    m.def("compute_tensor_min", &nectar::compute_tensor_min, 
        py::arg("device_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Computes the minimum value of a given tensor.");

    m.def("compute_tensor_max", &nectar::compute_tensor_max, 
        py::arg("device_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Computes the maximum value of a given tensor.");

    m.def("compute_tensor_range", &nectar::compute_tensor_range, 
        py::arg("device_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Computes the min-max range of a given tensor's values.");
}

