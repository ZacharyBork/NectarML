#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t gather(
        uintptr_t data_ptr,
        std::vector<int> data_shape,
        uintptr_t indices_ptr,
        std::vector<int> indices_shape,
        int dim,
        DType dtype
    );

    uintptr_t scatter(
        uintptr_t input_ptr, std::vector<int> input_shape,
        uintptr_t source_ptr, std::vector<int> source_shape,
        uintptr_t indices_ptr, std::vector<int> indices_shape,
        int dim, DType dtype
    );
}

void register_indexing(py::module_& m) {
    
    m.def("gather", &nectar::gather, 
        py::arg("data_ptr"),
        py::arg("data_shape"),
        py::arg("indices_ptr"),
        py::arg("indices_shape"),
        py::arg("dim"),
        py::arg("dtype"),
        "Takes values from input tensor by matching input indices.");

    m.def("scatter", &nectar::scatter, 
        py::arg("input_ptr"), py::arg("input_shape"),
        py::arg("src_ptr"), py::arg("src_shape"),
        py::arg("indices_ptr"), py::arg("indices_shape"),
        py::arg("dim"), py::arg("dtype"),
        "");

}
