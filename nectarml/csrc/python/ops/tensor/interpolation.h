#pragma once

#include "include/common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {

    /* Nearest Neighbor */

    uintptr_t upsample_nearest_1d(
        uintptr_t input_ptr,
        int B, int C, int L_in, int L_out,
        DType dtype
    );

    uintptr_t upsample_nearest_1d_backward(
        uintptr_t grad_ptr,
        int B, int C, int L_in, int L_out,
        DType dtype
    );

    uintptr_t upsample_nearest_2d(
        uintptr_t input_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        DType dtype
    );

    uintptr_t upsample_nearest_2d_backward(
        uintptr_t grad_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        DType dtype
    );

    uintptr_t upsample_nearest_3d(
        uintptr_t input_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        DType dtype
    );

    uintptr_t upsample_nearest_3d_backward(
        uintptr_t grad_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        DType dtype
    );

    /* Linear/Bilinear/Trilinear */

    uintptr_t upsample_linear(
        uintptr_t input_ptr,
        int B, int C, int L_in, int L_out,
        bool align_corners,
        DType dtype
    );

    uintptr_t upsample_linear_backward(
        uintptr_t grad_ptr,
        int B, int C, int L_in, int L_out,
        bool align_corners,
        DType dtype
    );

    uintptr_t upsample_bilinear(
        uintptr_t input_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        bool align_corners,
        DType dtype
    );

    uintptr_t upsample_bilinear_backward(
        uintptr_t grad_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        bool align_corners,
        DType dtype
    );

    uintptr_t upsample_trilinear(
        uintptr_t input_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        bool align_corners,
        DType dtype
    );

    uintptr_t upsample_trilinear_backward(
        uintptr_t grad_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        bool align_corners,
        DType dtype
    );

    /* CUBIC */

    uintptr_t upsample_bicubic(
        uintptr_t input_ptr,
        int B, int C, 
        int H_in, int W_in, 
        int H_out, int W_out,
        float a, bool align_corners,
        DType dtype
    );

    uintptr_t upsample_bicubic_backward(
        uintptr_t grad_ptr,
        int B, int C, 
        int H_in, int W_in, 
        int H_out, int W_out,
        float a, bool align_corners,
        DType dtype
    );

}

void register_interpolation(py::module_& m) {

    auto m_interpolation = m.def_submodule("interpolation", "Tensor interpolation submodule.");

    /* Nearest Neighbor */

    m_interpolation.def("upsample_nearest_1d", &nectar::upsample_nearest_1d, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("L_in"), py::arg("L_out"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_nearest_1d_backward", &nectar::upsample_nearest_1d_backward, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("L_in"), py::arg("L_out"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_nearest_2d", &nectar::upsample_nearest_2d, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_nearest_2d_backward", &nectar::upsample_nearest_2d_backward, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_nearest_3d", &nectar::upsample_nearest_3d, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_nearest_3d", &nectar::upsample_nearest_3d, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    /* Linear/Bilinear/Trilinear */

    m_interpolation.def("upsample_linear", &nectar::upsample_linear, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("L_in"), py::arg("L_out"),
        py::arg("align_corners"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_linear_backward", &nectar::upsample_linear_backward, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("L_in"), py::arg("L_out"),
        py::arg("align_corners"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_bilinear", &nectar::upsample_bilinear, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("align_corners"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_bilinear_backward", &nectar::upsample_bilinear_backward, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("align_corners"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_trilinear", &nectar::upsample_trilinear, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"),
        py::arg("align_corners"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_trilinear_backward", &nectar::upsample_trilinear_backward, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"),
        py::arg("align_corners"),
        py::arg("dtype"),
        "");

    /* CUBIC */
    
    m_interpolation.def("upsample_bicubic", &nectar::upsample_bicubic, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("a"), py::arg("align_corners"),
        py::arg("dtype"),
        "");

    m_interpolation.def("upsample_bicubic_backward", &nectar::upsample_bicubic_backward, 
        py::arg("grad_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("a"), py::arg("align_corners"),
        py::arg("dtype"),
        "");

}

