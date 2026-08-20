#pragma once

#include "include/common/dtype.h"

namespace py = pybind11;

namespace nectar {
    bool is_inf(uintptr_t in_ptr, size_t n_elements, DType dtype);
    bool is_finite(uintptr_t in_ptr, size_t n_elements, DType dtype);
    bool is_nan(uintptr_t in_ptr, size_t n_elements, DType dtype);
    bool has_inf(uintptr_t in_ptr, size_t n_elements, DType dtype);
    bool has_nan(uintptr_t in_ptr, size_t n_elements, DType dtype);
}

void register_inspection(py::module_& m) {
    
    m.def("is_inf", &nectar::is_inf, 
        py::arg("in_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Returns true if all values in Tensor's data are infinite, else returns false.");

    m.def("is_finite", &nectar::is_finite, 
        py::arg("in_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Returns true if all values in Tensor's data are finite, else false");

    m.def("is_nan", &nectar::is_nan, 
        py::arg("in_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Returns true if all values in Tensor's data are NaN, else false.");

    m.def("has_inf", &nectar::has_inf, 
        py::arg("in_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Returns true if any value in Tensor's data is infinite, else returns false.");

    m.def("has_nan", &nectar::has_nan, 
        py::arg("in_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Returns true if any value in Tensor's data is NaN, else returns false.");

}

