#pragma once

#include "common/dtype.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t concatenate(
        py::list ptrs, 
        std::vector<std::vector<int>> shapes, 
        int dim, 
        DType dtype
    );
}

void register_combination(py::module_& m) {
    
    auto m_combination = m.def_submodule("combination", "Tensor combination submodule.");
    
    m_combination.def("concatenate", &nectar::concatenate, 
        py::arg("ptrs"), 
        py::arg("shapes"), 
        py::arg("dim"),
        py::arg("dtype"),
        "Concatenates tensor data along a given dimension.");

}

