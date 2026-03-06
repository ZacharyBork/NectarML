#include "common.h"
#include "bindings/memory.h"
#include "bindings/device.h"
#include "bindings/elementwise.h"
#include "bindings/vision.h"
#include "bindings/reductions.h"
#include "bindings/combination.h"


namespace py = pybind11;

PYBIND11_MODULE(_nectarml, m) {
    m.doc() = "NectarML C++ extension module";

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("UInt8",   DType::UInt8)
        .value("Int32",   DType::Int32)
        .value("Bool",    DType::Bool);

    register_memory(m);
    register_device(m);
    register_elementwise(m);    
    register_vision(m);
    register_reductions(m);
    register_combination(m);
}
