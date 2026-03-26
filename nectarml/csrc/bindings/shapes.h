#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t permute(uintptr_t in_ptr, std::vector<int> shape, std::vector<int> dims, DType dtype);
    uintptr_t expand(uintptr_t in_ptr, std::vector<int> in_shape, std::vector<int> target_shape, DType dtype);
}

void register_shapes(py::module_& m) {
    
    auto m_shapes = m.def_submodule("shapes", "Tensor shapes submodule.");

    m_shapes.def("permute", &nectar::permute, 
        py::arg("in_ptr"), 
        py::arg("shape"), 
        py::arg("dims"),
        py::arg("dtype"),
        "Permutes tensor dimensions and returns as new tensor data.");

    m_shapes.def("expand", &nectar::expand, 
        py::arg("in_ptr"), 
        py::arg("in_shape"), 
        py::arg("target_shape"),
        py::arg("dtype"),
        "Expands input tensor's shape to match given target_shape.");

}

