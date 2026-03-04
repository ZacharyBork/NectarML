#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t reduce_sum(uintptr_t in_ptr, size_t n_elements, DType dtype);
}

void register_reductions(py::module_& m) {

    m.def("reduce_sum", &nectar::reduce_sum, 
        py::arg("ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Frees GPU memory of object at at given pointer.");

}

