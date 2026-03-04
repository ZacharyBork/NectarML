#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t reduce_sum(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        DType dtype
    );
}

void register_reductions(py::module_& m) {

    m.def("reduce_sum", &nectar::reduce_sum, 
        py::arg("in_ptr"),
        py::arg("shape"),
        py::arg("reduce_dim"),
        py::arg("dtype"),
        "Performs a dimension-wise sum reduction on a CUDA tensor.");

}

