#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t concatenate(py::list ptrs, std::vector<std::vector<int>> shapes, int dim, DType dtype);
    uintptr_t stack(py::list ptrs, std::vector<std::vector<int>> shapes, int dim, DType dtype);
}

void register_combination(py::module_& m) {
    
    m.def("concatenate", &nectar::concatenate, 
        py::arg("ptrs"), 
        py::arg("shapes"), 
        py::arg("dim"),
        py::arg("dtype"),
        "Concatenates tensor data along a given dimension.");

    m.def("stack", &nectar::stack, 
        py::arg("ptrs"), 
        py::arg("shapes"), 
        py::arg("dim"),
        py::arg("dtype"),
        "Concatenates tensor data along a given dimension.");

}

