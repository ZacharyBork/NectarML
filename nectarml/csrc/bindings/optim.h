#pragma once

#include "common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {

    uintptr_t sgd_update(
        uintptr_t param_ptr,
        uintptr_t grad_ptr,
        uintptr_t velocity_ptr,
        float     lr,
        float     momentum,
        float     dampening,
        float     weight_decay,
        bool      nesterov,
        bool      maximize,
        int       n_elements
    );

    uintptr_t adam_update(
        uintptr_t param_ptr,
        uintptr_t grad_ptr,
        uintptr_t exp_avg_ptr,
        uintptr_t exp_avg_sq_ptr,
        uintptr_t max_ea_sq_ptr,
        float     lr,
        float     beta1,
        float     beta2,
        float     eps,
        float     bias_correction1,
        float     bias_correction2,
        float     weight_decay,
        bool      decoupled_weight_decay,
        bool      amsgrad,
        bool      maximize,
        int       n_elements
    );

}

void register_optim(py::module_& m) {
    
    auto m_optim = m.def_submodule("optim", "Optimizer module.");
    
    m_optim.def("sgd_update", &nectar::sgd_update, 
        py::arg("param_ptr"), py::arg("grad_ptr"), py::arg("velocity_ptr"), 
        py::arg("lr"), py::arg("momentum"), py::arg("dampening"), 
        py::arg("weight_decay"), py::arg("nesterov"), py::arg("maximize"), 
        py::arg("n_elements"),
        "Fused update function for SGD optimizer.");

    m_optim.def("adam_update", &nectar::adam_update, 
        py::arg("param_ptr"), py::arg("grad_ptr"), 
        py::arg("exp_avg_ptr"), py::arg("exp_avg_sq_ptr"), 
        py::arg("max_ea_sq_ptr"), py::arg("lr"), 
        py::arg("beta1"), py::arg("beta2"), py::arg("eps"), 
        py::arg("bias_correction1"), py::arg("bias_correction2"),
        py::arg("weight_decay"), py::arg("decoupled_weight_decay"), 
        py::arg("amsgrad"), py::arg("maximize"), py::arg("n_elements"),
        "Fused update function for Adam optimizer.");

}

