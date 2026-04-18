#pragma once

#include "common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nectar {
    
    uintptr_t im2col_1d(
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, 
        int groups,
        DType dtype
    );

    uintptr_t col2im_1d(
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, 
        int groups,
        DType dtype
    );

    uintptr_t im2col_2d(
        uintptr_t input_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    );

    uintptr_t col2im_2d(
        uintptr_t input_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    );

}

void register_im2col(py::module_& m) {
    
    auto m_im2col = m.def_submodule("im2col", "Im2col submodule.");

    m_im2col.def("im2col_1d", &nectar::im2col_1d, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C_in"), py::arg("L"), 
        py::arg("C_out"), py::arg("K"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"),
        py::arg("groups"), py::arg("dtype"),
        "");

    m_im2col.def("col2im_1d", &nectar::col2im_1d, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C_in"), py::arg("L"), 
        py::arg("C_out"), py::arg("K"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"),
        py::arg("groups"), py::arg("dtype"),
        "");

    m_im2col.def("im2col_2d", &nectar::im2col_2d, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C_in"),
        py::arg("H"), py::arg("W"),
        py::arg("C_out"), py::arg("KH"), py::arg("KW"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("padding_h"), py::arg("padding_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("dtype"),
        "");

    m_im2col.def("col2im_2d", &nectar::col2im_2d, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C_in"),
        py::arg("H"), py::arg("W"),
        py::arg("C_out"), py::arg("KH"), py::arg("KW"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("padding_h"), py::arg("padding_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("dtype"),
        "");

}

