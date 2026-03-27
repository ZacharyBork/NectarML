#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);
    
    uintptr_t less_than(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t less_than_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);
    
    uintptr_t less_than_or_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t less_than_or_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);
   
    uintptr_t greater_than(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t greater_than_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);
    
    uintptr_t greater_than_or_equal(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t greater_than_or_equal_ts(uintptr_t in_ptr, float value, size_t n_elements, DType dtype);

    uintptr_t add(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t add_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t subtract(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t subtract_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t multiply(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t multiply_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t divide(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t divide_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
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
    uintptr_t atan2(
        uintptr_t b_ptr, uintptr_t a_ptr, 
        std::vector<int> b_shape,
        std::vector<int> a_shape,
        std::vector<int> out_shape,
        DType dtype
    );

    uintptr_t pow(uintptr_t base_ptr, float exponent, size_t n_elements, DType dtype);

    uintptr_t abs(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t floor(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t ceil(uintptr_t x_ptr, size_t n_elements, DType dtype);
    uintptr_t round(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t fmod(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    
    uintptr_t fmod_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );

    uintptr_t min(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t min_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t max(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t max_ts(
        uintptr_t in_ptr, 
        float value, 
        size_t n_elements, 
        DType dtype
    );
    uintptr_t clamp(uintptr_t base_ptr, float min_value, float max_value, size_t n_elements, DType dtype);

    uintptr_t sign(uintptr_t x_ptr, size_t n_elements, DType dtype);

    uintptr_t copysign(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );

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

    uintptr_t eq_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t lt_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t le_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t gt_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
    uintptr_t ge_mask_tensor(
        uintptr_t a_ptr, uintptr_t b_ptr,
        std::vector<int> a_shape,
        std::vector<int> b_shape,
        std::vector<int> out_shape,
        DType dtype
    );
}

void register_elementwise(py::module_& m) {

    auto m_elementwise = m.def_submodule("elementwise", "Elementwise submodule.");

    /* COMPARISON */

    m_elementwise.def("equal", &nectar::equal, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Elementwise eq (a==b). Returns boolean tensor data.");

    m_elementwise.def("equal_ts", &nectar::equal_ts, 
        py::arg("in_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise tensor-scalar eq (in_tensor==value). Returns boolean tensor data.");

    m_elementwise.def("less_than", &nectar::less_than, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Elementwise lt (a<b). Returns boolean tensor data.");

    m_elementwise.def("less_than_ts", &nectar::less_than_ts, 
        py::arg("in_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise tensor-scalar lt (in_tensor<value). Returns boolean tensor data.");

    m_elementwise.def("less_than_or_equal", &nectar::less_than_or_equal, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Elementwise le (a<=b). Returns boolean tensor data.");

    m_elementwise.def("less_than_or_equal_ts", &nectar::less_than_or_equal_ts, 
        py::arg("in_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise tensor-scalar le (in_tensor<=value). Returns boolean tensor data.");

    m_elementwise.def("greater_than", &nectar::greater_than, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Elementwise gt (a>b). Returns boolean tensor data.");

    m_elementwise.def("greater_than_ts", &nectar::greater_than_ts, 
        py::arg("in_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise tensor-scalar le (in_tensor>value). Returns boolean tensor data.");

    m_elementwise.def("greater_than_or_equal", &nectar::greater_than_or_equal, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Elementwise ge (a>=b). Returns boolean tensor data.");

    m_elementwise.def("greater_than_or_equal_ts", &nectar::greater_than_or_equal_ts, 
        py::arg("in_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Elementwise tensor-scalar le (in_tensor>=value). Returns boolean tensor data.");

    /* BASIC */

    m_elementwise.def("add", &nectar::add, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Adds tensor data a and b, then returns as new tensor data.");

    m_elementwise.def("add_ts", &nectar::add_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Adds tensor data a and scalar value, then returns as new tensor data.");

    m_elementwise.def("subtract", &nectar::subtract, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Subtracts tensor data b from a, then returns as new tensor data.");

    m_elementwise.def("subtract_ts", &nectar::subtract_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Subtracts scalar value from tensor a, then returns as new tensor data.");

    m_elementwise.def("multiply", &nectar::multiply, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Multiplies tensor data a by b, then returns as new tensor data.");

    m_elementwise.def("multiply_ts", &nectar::multiply_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Mutiplies tensor a by scalar value, then returns as new tensor data.");

    m_elementwise.def("divide", &nectar::divide, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Divides tensor data a by b, then returns as new tensor data.");

    m_elementwise.def("divide_ts", &nectar::divide_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Divides tensor a by scalar value, then returns as new tensor data.");

    m_elementwise.def("negate", &nectar::negate, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Negates tensor data x and returns as new tensor data.");

    /* SQRT */

    m_elementwise.def("sqrt", &nectar::sqrt, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the square root of tensor data x and returns as new tensor data.");

    m_elementwise.def("rsqrt", &nectar::rsqrt, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the reciprocal square root of tensor data x and returns as new tensor data.");

    /* EXPONENT */

    m_elementwise.def("exp", &nectar::exp, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the exponent of tensor data x and returns as new tensor data.");

    /* LOG */

    m_elementwise.def("log", &nectar::log, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the logarithm of tensor data x and returns as new tensor data.");

    m_elementwise.def("log2", &nectar::log2, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the logarithm^2 of tensor data x and returns as new tensor data.");

    m_elementwise.def("log10", &nectar::log10, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the logarithm^10 of tensor data x and returns as new tensor data.");
    
    /* SIN / COS */

    m_elementwise.def("sin", &nectar::sin, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the sine of tensor data x and returns as new tensor data.");

    m_elementwise.def("asin", &nectar::asin, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arc sine of tensor data x and returns as new tensor data.");

    m_elementwise.def("sinh", &nectar::sinh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the hyperbolic sine of tensor data x and returns as new tensor data.");

    m_elementwise.def("asinh", &nectar::asinh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the inverse hyperbolic sine of tensor data x and returns as new tensor data.");

    m_elementwise.def("cos", &nectar::cos, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the cosine of tensor data x and returns as new tensor data.");

    m_elementwise.def("acos", &nectar::acos, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arc cosine of tensor data x and returns as new tensor data.");

    m_elementwise.def("cosh", &nectar::cosh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the hyperbolic cosine of tensor data x and returns as new tensor data.");

    m_elementwise.def("acosh", &nectar::acosh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the non-negative inverse hyperbolic cosine of tensor data x and returns as new tensor data.");

    /* TAN / ATAN */

    m_elementwise.def("tan", &nectar::tan, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the tangent of tensor data x and returns as new tensor data.");

    m_elementwise.def("tanh", &nectar::tanh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the hyperbolic tangent of tensor data x and returns as new tensor data.");

    m_elementwise.def("atan", &nectar::atan, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the arctangent of tensor data x and returns as new tensor data.");

    m_elementwise.def("atanh", &nectar::atanh, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the inverse hyperbolid tangent of tensor data x and returns as new tensor data.");

    m_elementwise.def("atan2", &nectar::atan2, 
        py::arg("b_ptr"), py::arg("a_ptr"), 
        py::arg("b_shape"), py::arg("a_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Calculates the arctangent of the ratio of y and x and returns as new tensor data.");

    /* POW */

    m_elementwise.def("pow", &nectar::pow, 
        py::arg("x_ptr"),
        py::arg("exponent"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the value of x to the power of exponent and returns as new tensor data.");

    /* ABS */

    m_elementwise.def("abs", &nectar::abs, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the absolute of tensor data x and returns as new tensor data.");

    /* ROUNDING */

    m_elementwise.def("floor", &nectar::floor, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the largest integer less than or equal to x and returns as new tensor data.");

    m_elementwise.def("ceil", &nectar::ceil, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the largest integer greater than or equal to x and returns as new tensor data.");

    m_elementwise.def("round", &nectar::round, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Calculates the nearest integer to x and returns as new tensor data.");

    /* MODULO */

    // m_elementwise.def("mod", &nectar::mod, 
    //     py::arg("x_ptr"),
    //     py::arg("y_ptr"),
    //     py::arg("n_elements"),
    //     py::arg("dtype"),
    //     "Calculates integer modulo of x and y and returns as new tensor data.");

    m_elementwise.def("fmod", &nectar::fmod, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Calculates floating point modulo of x and y and returns as new tensor data.");

    m_elementwise.def("fmod_ts", &nectar::fmod_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Calculates the floating point modulo of input tensor and scalar value, then returns as new tensor data.");

    /* MIN / MAX */

    m_elementwise.def("min", &nectar::min, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Calculates the minimum of x and y and returns as new tensor data.");

    m_elementwise.def("min_ts", &nectar::min_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Calculates the minimum of input tensor data and scalar value and returns as new tensor data.");

    m_elementwise.def("max", &nectar::max, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Calculates the maximum of x and y and returns as new tensor data.");

    m_elementwise.def("max_ts", &nectar::max_ts, 
        py::arg("in_ptr"), 
        py::arg("value"),
        py::arg("n_elements"), 
        py::arg("dtype"),
        "Calculates the maximum of input tensor data and scalar value and returns as new tensor data.");

    /* SIGN */

    m_elementwise.def("sign", &nectar::sign, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns new tensor data with value -1 if x < 0.0, else 0.0 if x == 0.0, else 1.0.");

    /* COPYSIGN */

    m_elementwise.def("copysign", &nectar::copysign, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Returns new tensor data with magnitude of x and sign of y.");

    /* TRUNCATE */

    m_elementwise.def("trunc", &nectar::trunc, 
        py::arg("x_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Truncates data x and returns as new tensor data.");

    /* TENSOR/SCALAR OPS */

    m_elementwise.def("scalaradd", &nectar::scalaradd, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Adds scalar value from all elements of tensor data.");

    m_elementwise.def("scalarsub", &nectar::scalarsub, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Subtracts scalar value from all elements of tensor data.");

    m_elementwise.def("scalarmul", &nectar::scalarmul, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Multiplies all elements of tensor data by scalar value.");

    m_elementwise.def("scalardiv", &nectar::scalardiv, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Divides all elements of tensor data by scalar value.");

    m_elementwise.def("scalarmin", &nectar::scalarmin, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Takes the minimum value of all elements of tensor data and a given scalar value.");

    m_elementwise.def("scalarmax", &nectar::scalarmax, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Takes the maximum value of all elements of tensor data and a given scalar value.");

    /* MASKING */

    m_elementwise.def("eq_mask_scalar", &nectar::eq_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data == value, otherwise 0.0.");

    m_elementwise.def("lt_mask_scalar", &nectar::lt_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data < value, otherwise 0.0.");

    m_elementwise.def("le_mask_scalar", &nectar::le_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data <= value, otherwise 0.0.");

    m_elementwise.def("gt_mask_scalar", &nectar::gt_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data > value, otherwise 0.0.");

    m_elementwise.def("ge_mask_scalar", &nectar::ge_mask_scalar, 
        py::arg("base_ptr"),
        py::arg("value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data >= value, otherwise 0.0.");

    m_elementwise.def("eq_mask_tensor", &nectar::eq_mask_tensor, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x == tensor data y, otherwise 0.0.");

    m_elementwise.def("lt_mask_tensor", &nectar::lt_mask_tensor, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x < tensor data y, otherwise 0.0.");

    m_elementwise.def("le_mask_tensor", &nectar::le_mask_tensor, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x <= tensor data y, otherwise 0.0.");

    m_elementwise.def("gt_mask_tensor", &nectar::gt_mask_tensor, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x > tensor data y, otherwise 0.0.");

    m_elementwise.def("ge_mask_tensor", &nectar::ge_mask_tensor, 
        py::arg("a_ptr"), py::arg("b_ptr"),
        py::arg("a_shape"), py::arg("b_shape"), py::arg("out_shape"),
        py::arg("dtype"),
        "Returns mask with value 1.0 where tensor data x >= tensor data y, otherwise 0.0.");

    /* CLAMP */

    m_elementwise.def("clamp", &nectar::clamp, 
        py::arg("base_ptr"),
        py::arg("min_value"),
        py::arg("max_value"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Clamps tensor values between given min and max value.");

}

