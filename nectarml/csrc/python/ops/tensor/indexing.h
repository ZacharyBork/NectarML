#pragma once

#include "include/common/dtype.h"
#include <pybind11/pybind11.h>

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

    uintptr_t scatter_add(
        uintptr_t input_ptr, std::vector<int> input_shape,
        uintptr_t source_ptr, std::vector<int> source_shape,
        uintptr_t indices_ptr, std::vector<int> indices_shape,
        int dim, DType dtype
    );

    uintptr_t slice(
        uintptr_t input_ptr,
        std::vector<int> input_shape,
        std::vector<int> start,
        std::vector<int> count,
        std::vector<int> step,
        DType dtype
    );

    uintptr_t index_put(
        uintptr_t input_ptr,
        std::vector<int> input_shape,
        uintptr_t src_ptr,
        std::vector<int> start,
        std::vector<int> count,
        std::vector<int> step,
        DType dtype
    );
}

void register_indexing(py::module_& m) {
    
    auto m_indexing = m.def_submodule("indexing", "Tensor indexing submodule.");

    m_indexing.def("gather", &nectar::gather, 
        py::arg("data_ptr"),
        py::arg("data_shape"),
        py::arg("indices_ptr"),
        py::arg("indices_shape"),
        py::arg("dim"),
        py::arg("dtype"),
        "Takes values from input tensor by matching input indices.");

    m_indexing.def("scatter", &nectar::scatter, 
        py::arg("input_ptr"), py::arg("input_shape"),
        py::arg("src_ptr"), py::arg("src_shape"),
        py::arg("indices_ptr"), py::arg("indices_shape"),
        py::arg("dim"), py::arg("dtype"),
        "");

    m_indexing.def("scatter_add", &nectar::scatter_add, 
        py::arg("input_ptr"), py::arg("input_shape"),
        py::arg("src_ptr"), py::arg("src_shape"),
        py::arg("indices_ptr"), py::arg("indices_shape"),
        py::arg("dim"), py::arg("dtype"),
        "");

    m_indexing.def("slice", &nectar::slice, 
        py::arg("input_ptr"), py::arg("input_shape"),
        py::arg("start"), py::arg("count"), py::arg("step"),
        py::arg("dtype"),
        "");

    m_indexing.def("index_put", &nectar::index_put, 
        py::arg("input_ptr"), py::arg("input_shape"), py::arg("src_ptr"), 
        py::arg("start"), py::arg("count"), py::arg("step"),
        py::arg("dtype"),
        "");

}
