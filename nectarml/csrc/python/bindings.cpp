#include "include/common/dtype.h"

#include "pool.h"

#include "ops/amp.h"
#include "ops/system.h"
#include "ops/tensor.h"
#include "ops/optim.h"
#include "ops/vision.h"

namespace py = pybind11;

void destroy_cublas_handle();

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

    register_allocator_pool(m);
    register_im2col(m);
    register_memory(m);
    register_device(m);
    register_utils(m);
    register_vision(m);
    register_inspection(m);

    register_amp(m);
    register_optim(m);

    auto m_tensor = m.def_submodule("tensor", "Tensor submodule.");
    register_combination(m_tensor);
    register_conv(m_tensor);
    register_elementwise(m_tensor);
    register_indexing(m_tensor);
    register_interpolation(m_tensor);
    register_matmul(m_tensor);
    register_norm(m_tensor);
    register_padding(m_tensor);
    register_pooling(m_tensor);
    register_reductions(m_tensor);
    register_shapes(m_tensor);
    register_sorting(m_tensor);

}
