#pragma once

#include "include/common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {
    
    uintptr_t batch_norm_forward(
        uintptr_t x_ptr,
        uintptr_t gamma_ptr,
        uintptr_t beta_ptr,
        uintptr_t mean_ptr,
        uintptr_t var_ptr,
        int N, int C, int H, int W,
        bool reduce_N, bool reduce_H, bool reduce_W,
        float eps);

    void batch_norm_backward(
        uintptr_t grad_out_ptr,
        uintptr_t x_ptr,
        uintptr_t mean_ptr,
        uintptr_t var_ptr,
        uintptr_t gamma_ptr,
        uintptr_t dx_ptr,
        uintptr_t dgamma_ptr,
        uintptr_t dbeta_ptr,
        int N, int C, int H, int W,
        bool reduce_N, bool reduce_H, bool reduce_W,
        float eps
    );
}

void register_norm(py::module_& m) {
    
    auto m_norm = m.def_submodule("norm", "Tensor normalization submodule.");

    m_norm.def("batch_norm_forward", &nectar::batch_norm_forward, 
        py::arg("x_ptr"), py::arg("gamma_ptr"), py::arg("beta_ptr"),
        py::arg("mean_ptr"), py::arg("var_ptr"),
        py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("reduce_N"), py::arg("reduce_H"), py::arg("reduce_W"),
        py::arg("eps"),
        "");

    m_norm.def("batch_norm_backward", &nectar::batch_norm_backward, 
        py::arg("grad_out_ptr"), py::arg("x_ptr"),
        py::arg("mean_ptr"), py::arg("var_ptr"), py::arg("gamma_ptr"),
        py::arg("dx_ptr"), py::arg("dgamma_ptr"), py::arg("dbeta_ptr"),
        py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("reduce_N"), py::arg("reduce_H"), py::arg("reduce_W"),
        py::arg("eps"), 
        "");

}
