#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ops/common.h>

namespace py = pybind11;

/* MEMORY */

void free_cuda(uintptr_t ptr);
uintptr_t alloc_cuda_full(size_t n_elements, DType dtype, double fill_value);

/* DEVICE */

uintptr_t cast_tensor(uintptr_t device_ptr, size_t n_elements, DType src_dtype, DType dst_dtype);
uintptr_t to_cuda(uintptr_t host_ptr, size_t n_elements, DType dtype);
py::array to_cpu(uintptr_t device_ptr, std::vector<ssize_t> shape, DType dtype);

/* ELEMENTWISE */

uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);

/* VISION.TRANSFORMS */

py::array_t<uint8_t> hue_shift(py::array_t<uint8_t> image, float shift);

PYBIND11_MODULE(_nectarml, m) {
    m.doc() = "NectarML C++ extension module";

    /* COMMON */

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("UInt8",   DType::UInt8)
        .value("Int32",   DType::Int32);

    /* MEMORY MANAGEMENT */

    m.def("free_cuda", &free_cuda, 
        py::arg("ptr"),
        "Frees GPU memory of object at at given pointer.");

    m.def("alloc_cuda_full", &alloc_cuda_full, 
        py::arg("n_elements"),
        py::arg("dtype"),
        py::arg("fill_value"),
        "Allocates CUDA memory for tensor data of n_elements, and fills with fill_value.");

    /* DEVICE MANAGEMENT */

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

    /* ELEMENTWISE */

    m.def("add", &add, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds tensor data a and b, then returns as new tensor data.");

    /* VISION.TRANSFORMS */

    m.def("hue_shift", &hue_shift,
        py::arg("image"),
        py::arg("shift"),
        "Shift the hue of an RGB image by the given number of degrees."
    );
}
