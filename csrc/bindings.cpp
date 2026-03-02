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

namespace nectar {
    uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t subtract(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t multiply(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t divide(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t sqrt(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t rsqrt(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t exp(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log2(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log10(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t sin(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t cos(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t tan(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atan2(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t pow(uintptr_t base_ptr, float exponent, size_t n_elements, DType dtype);
    uintptr_t abs(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t mod(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t fmod(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t min(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t max(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t copysign(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t trunc(uintptr_t x_ptr, size_t n_elements, DType dtype);
}

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

    m.def("add", &nectar::add, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds tensor data a and b, then returns as new tensor data.");

    m.def("subtract", &nectar::subtract, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds tensor data a and b, then returns as new tensor data.");

    m.def("multiply", &nectar::multiply, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds tensor data a and b, then returns as new tensor data.");

    m.def("divide", &nectar::divide, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds tensor data a and b, then returns as new tensor data.");

    m.def("sqrt", &nectar::sqrt, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the square root of tensor data x and returns as new tensor data.");

    m.def("rsqrt", &nectar::rsqrt, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the reciprocal square root of tensor data x and returns as new tensor data.");

    m.def("exp", &nectar::exp, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the exponent of tensor data x and returns as new tensor data.");

    m.def("log", &nectar::log, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the logarithm of tensor data x and returns as new tensor data.");

    m.def("log2", &nectar::log2, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the logarithm^2 of tensor data x and returns as new tensor data.");

    m.def("log10", &nectar::log10, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the logarithm^10 of tensor data x and returns as new tensor data.");

    m.def("sin", &nectar::sin, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the sine of tensor data x and returns as new tensor data.");

    m.def("cos", &nectar::cos, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the cosine of tensor data x and returns as new tensor data.");

    m.def("tan", &nectar::tan, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the tangent of tensor data x and returns as new tensor data.");

    m.def("atan", &nectar::atan, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arctangent of tensor data x and returns as new tensor data.");

    m.def("atan2", &nectar::atan2, 
        py::arg("y_ptr"),
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arctangent of the ratio of y and x and returns as new tensor data.");

    m.def("pow", &nectar::pow, 
        py::arg("x_ptr"),
        py::arg("exponent"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the value of x to the power of exponent and returns as new tensor data.");

    m.def("abs", &nectar::abs, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the absolute of tensor data x and returns as new tensor data.");

    m.def("floor", &nectar::floor, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the largest integer less than or equal to x and returns as new tensor data.");

    m.def("ceil", &nectar::ceil, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the largest integer greater than or equal to x and returns as new tensor data.");

    m.def("round", &nectar::round, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the nearest integer to x and returns as new tensor data.");

    m.def("mod", &nectar::mod, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates integer modulo of x and y and returns as new tensor data.");

    m.def("fmod", &nectar::fmod, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates floating point modulo of x and y and returns as new tensor data.");

    m.def("min", &nectar::min, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the minimum of x and y and returns as new tensor data.");

    m.def("max", &nectar::max, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the maximum of x and y and returns as new tensor data.");

    m.def("copysign", &nectar::copysign, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns new tensor data with magnitude of x and sign of y.");

    m.def("trunc", &nectar::trunc, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Truncates data x and returns as new tensor data.");

    /* VISION.TRANSFORMS */

    m.def("hue_shift", &hue_shift,
        py::arg("image"),
        py::arg("shift"),
        "Shift the hue of an RGB image by the given number of degrees."
    );
}
