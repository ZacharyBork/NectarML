#pragma once

#include "common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {

    bool unscale_and_check_grad(
        uintptr_t grad_ptr,
        float     inv_scale,
        int       n_elements
    );  

}

void register_amp(py::module_& m) {
    
    auto m_amp = m.def_submodule("amp", "AMP utility module.");
    
    m_amp.def("unscale_and_check_grad", [](
            uintptr_t grad_ptr,
            float     inv_scale,
            int       n_elements
        ) { 
            return nectar::unscale_and_check_grad(grad_ptr, inv_scale, n_elements);
        }, 
        "Used by GradScaler. Unscales grad tensor and if result is finite."
    );

}

