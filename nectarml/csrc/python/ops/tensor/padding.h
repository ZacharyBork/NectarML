#pragma once

#include "include/common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {
    
    uintptr_t pad(
        uintptr_t          input_ptr,
        std::vector<int>   input_shape,
        std::vector<int>   pad_before,
        std::vector<int>   pad_after,
        const std::string& mode,
        float              constant_value,
        DType              dtype
    );

    void pad_backward(
        uintptr_t          grad_out_ptr,
        uintptr_t          grad_in_ptr,
        std::vector<int>   input_shape,
        std::vector<int>   pad_before,
        std::vector<int>   pad_after,
        const std::string& mode
    );

}

void register_padding(py::module_& m) {
    
    auto m_padding = m.def_submodule("padding", "Padding submodule.");

    m_padding.def("pad", &nectar::pad, 
        py::arg("input_ptr"), 
        py::arg("input_shape"), 
        py::arg("pad_before"),
        py::arg("pad_after"), 
        py::arg("mode"),
        py::arg("constant_value"),
        py::arg("dtype"),
        "");

    m_padding.def("pad_backward", &nectar::pad_backward, 
        py::arg("grad_out_ptr"), 
        py::arg("grad_in_ptr"), 
        py::arg("input_shape"), 
        py::arg("pad_before"),
        py::arg("pad_after"), 
        py::arg("mode"),
        "");

}

