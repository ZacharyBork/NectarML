#include "common.h"
#include "bindings/memory.h"
#include "bindings/device.h"
#include "bindings/utils.h"
#include "bindings/elementwise.h"
#include "bindings/vision.h"
#include "bindings/reductions.h"
#include "bindings/combination.h"
#include "bindings/shapes.h"
#include "bindings/matmul.h"
#include "bindings/indexing.h"
#include "bindings/inspection.h"
#include "bindings/conv.h"
#include "bindings/interpolation.h"
#include "bindings/padding.h"

namespace py = pybind11;

void destroy_cublas_handle();

namespace tensor { }

PYBIND11_MODULE(_nectarml, m) {
    m.doc() = "NectarML C++ extension module";

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("UInt8",   DType::UInt8)
        .value("Int32",   DType::Int32)
        .value("Bool",    DType::Bool);

    m.def("destroy_cublas_handle", &destroy_cublas_handle, 
        "Destroys cuBLAS handle. Registered atexit for Python module.");


    auto m_tensor = m.def_submodule("tensor", "Tensor submodule.");
    register_combination(m_tensor);
    register_conv(m_tensor);
    register_elementwise(m_tensor);
    register_indexing(m_tensor);
    register_interpolation(m_tensor);
    register_matmul(m_tensor);
    register_padding(m_tensor);
    register_reductions(m_tensor);
    register_shapes(m_tensor);


    register_memory(m);
    register_device(m);
    register_utils(m);
    register_vision(m);
    register_inspection(m);

}
