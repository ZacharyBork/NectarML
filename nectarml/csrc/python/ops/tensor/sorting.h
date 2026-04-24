#pragma once

#include "include/common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {

    std::pair<uintptr_t, uintptr_t> sort(
        uintptr_t input_ptr,
        int total,
        int dim_size,
        int outer,
        int inner,
        bool descending,
        DType dtype
    );

}

void register_sorting(py::module_& m) {
    
    auto m_sorting = m.def_submodule("sorting", "Tensor sorting submodule.");
    
    /* 1-Dimensional */

    m_sorting.def("sort", &nectar::sort, 
        py::arg("input_ptr"), 
        py::arg("total"), 
        py::arg("dim_size"),
        py::arg("outer"), py::arg("inner"), 
        py::arg("descending"), 
        py::arg("dtype"),
        "");

}

