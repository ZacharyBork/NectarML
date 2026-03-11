#include "common.h"

namespace py = pybind11;

uintptr_t cast_tensor(uintptr_t device_ptr, size_t n_elements, DType src_dtype, DType dst_dtype);
uintptr_t to_cuda(uintptr_t host_ptr, size_t n_elements, DType dtype);
py::array to_cpu(uintptr_t device_ptr, std::vector<size_t> shape, DType dtype);

void register_device(py::module_& m) {
    m.def("cast_tensor", &cast_tensor, 
        py::arg("device_ptr"), 
        py::arg("n_elements"), 
        py::arg("src_dtype"),
        py::arg("dst_dtype"),
        "Changes DType of CUDA tensor data.");

    m.def("to_cuda", &to_cuda, 
        py::arg("host_ptr"), 
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Moves tensor data from system memory to GPU memory.");

    m.def("to_cpu", &to_cpu, 
        py::arg("device_ptr"),
        py::arg("shape"),
        py::arg("dtype"),
        "Moves tensor data from GPU memory to system memory.");
}

