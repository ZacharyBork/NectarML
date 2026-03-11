#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t less_than(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t less_than_or_equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t greater_than(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t greater_than_or_equal(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    
    uintptr_t add(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t subtract(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t multiply(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t divide(uintptr_t a_ptr, uintptr_t b_ptr, size_t n_elements, DType dtype);
    uintptr_t negate(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t sqrt(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t rsqrt(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t exp(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t log(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log2(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t log10(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t sin(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t asin(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t sinh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t asinh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t cos(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t acos(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t cosh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t acosh(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t tan(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t tanh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atan(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atanh(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t atan2(uintptr_t y_ptr, uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t pow(uintptr_t base_ptr, float exponent, size_t n_elements, DType dtype);

    uintptr_t abs(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t fmod(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);

    uintptr_t min(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t max(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t clamp(uintptr_t base_ptr, float min_value, float max_value, size_t n_elements, DType dtype);

    uintptr_t sign(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t copysign(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);

    uintptr_t trunc(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t scalaradd(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarsub(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarmul(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalardiv(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarmin(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t scalarmax(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t eq_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t lt_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t le_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t gt_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);
    uintptr_t ge_mask_scalar(uintptr_t base_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t eq_mask_tensor(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t lt_mask_tensor(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t le_mask_tensor(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t gt_mask_tensor(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
    uintptr_t ge_mask_tensor(uintptr_t x_ptr, uintptr_t y_ptr, size_t n_elements, DType dtype);
}

void register_elementwise(py::module_& m) {

    /* COMPARISON */

    m.def("equal", &nectar::equal, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise eq (a==b). Returns boolean tensor data.");

    m.def("less_than", &nectar::less_than, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise lt (a<b). Returns boolean tensor data.");

    m.def("less_than_or_equal", &nectar::less_than_or_equal, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise le (a<=b). Returns boolean tensor data.");

    m.def("greater_than", &nectar::greater_than, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise gt (a>b). Returns boolean tensor data.");

    m.def("greater_than_or_equal", &nectar::greater_than_or_equal, 
        py::arg("a_ptr"),
        py::arg("b_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise ge (a>=b). Returns boolean tensor data.");

    /* BASIC */

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

    m.def("negate", &nectar::negate, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Negates tensor data x and returns as new tensor data.");

    /* SQRT */

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

    /* EXPONENT */

    m.def("exp", &nectar::exp, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the exponent of tensor data x and returns as new tensor data.");

    /* LOG */

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
    
    /* SIN / COS */

    m.def("sin", &nectar::sin, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the sine of tensor data x and returns as new tensor data.");

    m.def("asin", &nectar::asin, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arc sine of tensor data x and returns as new tensor data.");

    m.def("sinh", &nectar::sinh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the hyperbolic sine of tensor data x and returns as new tensor data.");

    m.def("asinh", &nectar::asinh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the inverse hyperbolic sine of tensor data x and returns as new tensor data.");

    m.def("cos", &nectar::cos, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the cosine of tensor data x and returns as new tensor data.");

    m.def("acos", &nectar::acos, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arc cosine of tensor data x and returns as new tensor data.");

    m.def("cosh", &nectar::cosh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the hyperbolic cosine of tensor data x and returns as new tensor data.");

    m.def("acosh", &nectar::acosh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the non-negative inverse hyperbolic cosine of tensor data x and returns as new tensor data.");

    /* TAN / ATAN */

    m.def("tan", &nectar::tan, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the tangent of tensor data x and returns as new tensor data.");

    m.def("tanh", &nectar::tanh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the hyperbolic tangent of tensor data x and returns as new tensor data.");

    m.def("atan", &nectar::atan, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arctangent of tensor data x and returns as new tensor data.");

    m.def("atanh", &nectar::atanh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the inverse hyperbolid tangent of tensor data x and returns as new tensor data.");

    m.def("atan2", &nectar::atan2, 
        py::arg("y_ptr"),
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arctangent of the ratio of y and x and returns as new tensor data.");

    /* POW */

    m.def("pow", &nectar::pow, 
        py::arg("x_ptr"),
        py::arg("exponent"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the value of x to the power of exponent and returns as new tensor data.");

    /* ABS */

    m.def("abs", &nectar::abs, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the absolute of tensor data x and returns as new tensor data.");

    /* ROUNDING */

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

    /* MODULO */

    // m.def("mod", &nectar::mod, 
    //     py::arg("x_ptr"),
    //     py::arg("y_ptr"),
    //     py::arg("n_elements"),
    //     py::arg("dtype"),
    //     "Calculates integer modulo of x and y and returns as new tensor data.");

    m.def("fmod", &nectar::fmod, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates floating point modulo of x and y and returns as new tensor data.");

    /* MIN / MAX */

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

    /* SIGN */

    m.def("sign", &nectar::sign, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns new tensor data with value -1 if x < 0.0, else 0.0 if x == 0.0, else 1.0.");

    /* COPYSIGN */

    m.def("copysign", &nectar::copysign, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns new tensor data with magnitude of x and sign of y.");

    /* TRUNCATE */

    m.def("trunc", &nectar::trunc, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Truncates data x and returns as new tensor data.");

    /* TENSOR/SCALAR OPS */

    m.def("scalaradd", &nectar::scalaradd, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds scalar value from all elements of tensor data.");

    m.def("scalarsub", &nectar::scalarsub, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Subtracts scalar value from all elements of tensor data.");

    m.def("scalarmul", &nectar::scalarmul, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Multiplies all elements of tensor data by scalar value.");

    m.def("scalardiv", &nectar::scalardiv, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Divides all elements of tensor data by scalar value.");

    m.def("scalarmin", &nectar::scalarmin, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Takes the minimum value of all elements of tensor data and a given scalar value.");

    m.def("scalarmax", &nectar::scalarmax, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Takes the maximum value of all elements of tensor data and a given scalar value.");

    /* MASKING */

    m.def("eq_mask_scalar", &nectar::eq_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data == value, otherwise 0.0.");

    m.def("lt_mask_scalar", &nectar::lt_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data < value, otherwise 0.0.");

    m.def("le_mask_scalar", &nectar::le_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data <= value, otherwise 0.0.");

    m.def("gt_mask_scalar", &nectar::gt_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data > value, otherwise 0.0.");

    m.def("ge_mask_scalar", &nectar::ge_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data >= value, otherwise 0.0.");

    m.def("eq_mask_tensor", &nectar::eq_mask_tensor, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x == tensor data y, otherwise 0.0.");

    m.def("lt_mask_tensor", &nectar::lt_mask_tensor, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x < tensor data y, otherwise 0.0.");

    m.def("le_mask_tensor", &nectar::le_mask_tensor, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x <= tensor data y, otherwise 0.0.");

    m.def("gt_mask_tensor", &nectar::gt_mask_tensor, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x > tensor data y, otherwise 0.0.");

    m.def("ge_mask_tensor", &nectar::ge_mask_tensor, 
        py::arg("x_ptr"),
        py::arg("y_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x >= tensor data y, otherwise 0.0.");

    /* CLAMP */

    m.def("clamp", &nectar::clamp, 
        py::arg("base_ptr"),
        py::arg("min_value"),
        py::arg("max_value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Clamps tensor values between given min and max value.");

}

