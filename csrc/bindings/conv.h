#include "common.h"

namespace py = pybind11;

namespace nectar {
    
    uintptr_t conv1d(
        uintptr_t input_ptr,
        uintptr_t weight_ptr,
        uintptr_t bias_ptr,
        int B, int C_in, int L,
        int C_out, int K,
        int stride, int padding, int dilation, int groups,
        DType dtype
    );

    uintptr_t conv1d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    );

    uintptr_t conv1d_backward_weight(
        uintptr_t out_grad_ptr,
        uintptr_t input_ptr,
        int B, int C_in, int L,
        int C_out, int K, int L_out,
        int stride, int padding, int dilation,
        DType dtype
    );

    uintptr_t conv2d(
        uintptr_t input_ptr,
        uintptr_t weight_ptr,
        uintptr_t bias_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        int groups,
        DType dtype
    );

    uintptr_t conv2d_backward_input(
        uintptr_t out_grad_ptr,
        uintptr_t weight_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    );

    uintptr_t conv2d_backward_weight(
        uintptr_t out_grad_ptr,
        uintptr_t input_ptr,
        int B, int C_in, int H, int W,
        int C_out, int KH, int KW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        DType dtype
    );
}

void register_conv(py::module_& m) {
    
    m.def("conv1d", &nectar::conv1d, 
        py::arg("input_ptr"), 
        py::arg("weight_ptr"), 
        py::arg("bias_ptr"),
        py::arg("B"), py::arg("C_in"), py::arg("L"),
        py::arg("C_out"), py::arg("K"),
        py::arg("stride"), py::arg("padding"), 
        py::arg("dilation"), py::arg("groups"),
        py::arg("dtype"),
        "");

    m.def("conv1d_backward_input", &nectar::conv1d_backward_input, 
        py::arg("out_grad_ptr"), 
        py::arg("weight_ptr"), 
        py::arg("B"), py::arg("C_in"), py::arg("L"),
        py::arg("C_out"), py::arg("K"), py::arg("L_out"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"), 
        py::arg("dtype"),
        "");

    m.def("conv1d_backward_weight", &nectar::conv1d_backward_weight, 
        py::arg("out_grad_ptr"), 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C_in"), py::arg("L"),
        py::arg("C_out"), py::arg("K"), py::arg("L_out"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"), 
        py::arg("dtype"),
        "");

    m.def("conv2d", &nectar::conv2d, 
        py::arg("input_ptr"), 
        py::arg("weight_ptr"), 
        py::arg("bias_ptr"),
        py::arg("B"), py::arg("C_in"), py::arg("H"), py::arg("W"),
        py::arg("C_out"), py::arg("KH"), py::arg("KW"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("padding_h"), py::arg("padding_w"), 
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("groups"),
        py::arg("dtype"),
        "");

    m.def("conv2d_backward_input", &nectar::conv2d_backward_input, 
        py::arg("out_grad_ptr"), 
        py::arg("weight_ptr"), 
        py::arg("B"), py::arg("C_in"), py::arg("H"), py::arg("W"),
        py::arg("C_out"), py::arg("KH"), py::arg("KW"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("padding_h"), py::arg("padding_w"), 
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("dtype"),
        "");

    m.def("conv2d_backward_weight", &nectar::conv2d_backward_weight, 
        py::arg("out_grad_ptr"), 
        py::arg("input_ptr"), 
        py::arg("B"), py::arg("C_in"), py::arg("H"), py::arg("W"),
        py::arg("C_out"), py::arg("KH"), py::arg("KW"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("padding_h"), py::arg("padding_w"), 
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("dtype"),
        "");

}

