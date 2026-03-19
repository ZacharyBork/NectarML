#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t matmul(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape, std::vector<int> b_shape,
        DType dtype
    );
}

void register_matmul(py::module_& m) {

    m.def("matmul", &nectar::matmul, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("a_shape"),
        py::arg("b_shape"),
        py::arg("dtype"),
        "Performs matrix multiplication between a and b tensor data.");

}

