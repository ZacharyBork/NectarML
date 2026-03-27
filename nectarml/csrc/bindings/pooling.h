#include "common.h"

namespace py = pybind11;

namespace nectar {
    
    /* AVERAGE POOL */

    uintptr_t avg_pool1d_forward(
        uintptr_t input_ptr,
        int B, int C, int L, int L_out,
        int K, int S, int P,
        bool count_include_pad,
        DType dtype
    );

    uintptr_t avg_pool1d_backward(
        uintptr_t out_grad_ptr,
        int B, int C, int L, int L_out,
        int K, int S, int P,
        bool count_include_pad,
        DType dtype
    );

    uintptr_t avg_pool2d_forward(
        uintptr_t input_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW,
        bool count_include_pad,
        DType dtype
    );

    uintptr_t avg_pool2d_backward(
        uintptr_t out_grad_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW,
        bool count_include_pad,
        DType dtype
    );

    uintptr_t avg_pool3d_forward(
        uintptr_t input_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        int KD, int KH, int KW,
        int SD, int SH, int SW,
        int PD, int PH, int PW,
        bool count_include_pad,
        DType dtype
    );

    uintptr_t avg_pool3d_backward(
        uintptr_t out_grad_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        int KD, int KH, int KW,
        int SD, int SH, int SW,
        int PD, int PH, int PW,
        bool count_include_pad,
        DType dtype
    );

    /* MAX POOL */

    std::pair<uintptr_t, uintptr_t> max_pool1d_forward(
        uintptr_t input_ptr,
        int B, int C, int L, int L_out,
        int K, int S, int P, int D,
        DType dtype
    );

    uintptr_t max_pool1d_backward(
        uintptr_t out_grad_ptr,
        uintptr_t indices_ptr,
        int B, int C, int L, int L_out,
        DType dtype
    );

    std::pair<uintptr_t, uintptr_t> max_pool2d_forward(
        uintptr_t input_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        int KH, int KW,
        int SH, int SW,
        int PH, int PW, int D,
        DType dtype
    );

    uintptr_t max_pool2d_backward(
        uintptr_t out_grad_ptr,
        uintptr_t indices_ptr,
        int B, int C, int H, int W,
        int H_out, int W_out,
        DType dtype
    );

    std::pair<uintptr_t, uintptr_t> max_pool3d_forward(
        uintptr_t input_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        int KD, int KH, int KW,
        int SD, int SH, int SW,
        int PD, int PH, int PW, int Dil,
        DType dtype
    );

    uintptr_t max_pool3d_backward(
        uintptr_t out_grad_ptr,
        uintptr_t indices_ptr,
        int B, int C, int D, int H, int W,
        int D_out, int H_out, int W_out,
        DType dtype
    );

}

void register_pooling(py::module_& m) {
    
    auto m_pooling = m.def_submodule("pooling", "Tensor pooling submodule.");
    
    /* AVERAGE POOL */

    m_pooling.def("avg_pool1d_forward", &nectar::avg_pool1d_forward, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("L"), py::arg("L_out"), 
        py::arg("K"), py::arg("S"), py::arg("P"), 
        py::arg("count_include_pad"), py::arg("dtype"),
        "");

    m_pooling.def("avg_pool1d_backward", &nectar::avg_pool1d_backward, 
        py::arg("out_grad_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("L"), py::arg("L_out"), 
        py::arg("K"), py::arg("S"), py::arg("P"), 
        py::arg("count_include_pad"), py::arg("dtype"),
        "");

    m_pooling.def("avg_pool2d_forward", &nectar::avg_pool2d_forward, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("H_out"), py::arg("W_out"), 
        py::arg("KH"), py::arg("KW"), 
        py::arg("SH"), py::arg("SW"), 
        py::arg("PH"), py::arg("PW"), 
        py::arg("count_include_pad"), py::arg("dtype"),
        "");

    m_pooling.def("avg_pool2d_backward", &nectar::avg_pool2d_backward, 
        py::arg("out_grad_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("H_out"), py::arg("W_out"), 
        py::arg("KH"), py::arg("KW"), 
        py::arg("SH"), py::arg("SW"), 
        py::arg("PH"), py::arg("PW"), 
        py::arg("count_include_pad"), py::arg("dtype"),
        "");

    m_pooling.def("avg_pool3d_forward", &nectar::avg_pool3d_forward, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("D"), py::arg("H"), py::arg("W"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"), 
        py::arg("KD"), py::arg("KH"), py::arg("KW"), 
        py::arg("SD"), py::arg("SH"), py::arg("SW"), 
        py::arg("PD"), py::arg("PH"), py::arg("PW"), 
        py::arg("count_include_pad"), py::arg("dtype"),
        "");

    m_pooling.def("avg_pool3d_backward", &nectar::avg_pool3d_backward, 
        py::arg("out_grad_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("D"), py::arg("H"), py::arg("W"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"), 
        py::arg("KD"), py::arg("KH"), py::arg("KW"), 
        py::arg("SD"), py::arg("SH"), py::arg("SW"), 
        py::arg("PD"), py::arg("PH"), py::arg("PW"), 
        py::arg("count_include_pad"), py::arg("dtype"),
        "");
    
    /* MAX POOL */

    m_pooling.def("max_pool1d_forward", &nectar::max_pool1d_forward, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("L"), py::arg("L_out"), 
        py::arg("K"), py::arg("S"), py::arg("P"), py::arg("D"), 
        py::arg("dtype"),
        "");

    m_pooling.def("max_pool1d_backward", &nectar::max_pool1d_backward, 
        py::arg("out_grad_ptr"), py::arg("indices_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("L"), py::arg("L_out"), 
        py::arg("dtype"),
        "");

    m_pooling.def("max_pool2d_forward", &nectar::max_pool2d_forward, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("H_out"), py::arg("W_out"), 
        py::arg("KH"), py::arg("KW"), 
        py::arg("SH"), py::arg("KW"), 
        py::arg("PH"), py::arg("KW"), 
        py::arg("D"), py::arg("dtype"),
        "");

    m_pooling.def("max_pool2d_backward", &nectar::max_pool2d_backward, 
        py::arg("out_grad_ptr"), py::arg("indices_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("H_out"), py::arg("W_out"), 
        py::arg("dtype"),
        "");

    m_pooling.def("max_pool3d_forward", &nectar::max_pool3d_forward, 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("D"), py::arg("H"), py::arg("W"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"), 
        py::arg("KD"), py::arg("KH"), py::arg("KW"), 
        py::arg("SD"), py::arg("SH"), py::arg("KW"), 
        py::arg("PD"), py::arg("PH"), py::arg("KW"), 
        py::arg("Dil"), py::arg("dtype"),
        "");

    m_pooling.def("max_pool3d_backward", &nectar::max_pool3d_backward, 
        py::arg("out_grad_ptr"), py::arg("indices_ptr"), 
        py::arg("B"), py::arg("C"), py::arg("D"), py::arg("H"), py::arg("W"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"), 
        py::arg("dtype"),
        "");

}

