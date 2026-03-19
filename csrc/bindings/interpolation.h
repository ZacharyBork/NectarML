#include "common.h"

namespace py = pybind11;

namespace nectar {

    /* Nearest Neighbor */

    uintptr_t upsample_nearest_1d(
        uintptr_t input_ptr,
        int B, int C, int L_in, int L_out,
        DType dtype
    );

    uintptr_t upsample_nearest_1d_backward(
        uintptr_t grad_output_ptr,
        int B, int C, int L_in, int L_out,
        DType dtype
    );

    uintptr_t upsample_nearest_2d(
        uintptr_t input_ptr,
        int B, int C, int H_in, int W_in, int H_out, int W_out,
        DType dtype
    );

    uintptr_t upsample_nearest_2d_backward(
        uintptr_t grad_output_ptr,
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
        uintptr_t grad_output_ptr,
        int B, int C,
        int D_in, int H_in, int W_in,
        int D_out, int H_out, int W_out,
        DType dtype
    );

}

void register_interpolation(py::module_& m) {

    /* Nearest Neighbor */

    m.def("upsample_nearest_1d", &nectar::upsample_nearest_1d, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("L_in"), py::arg("L_out"),
        py::arg("dtype"),
        "");

    m.def("upsample_nearest_1d_backward", &nectar::upsample_nearest_1d_backward, 
        py::arg("grad_output_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("L_in"), py::arg("L_out"),
        py::arg("dtype"),
        "");

    m.def("upsample_nearest_2d", &nectar::upsample_nearest_2d, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    m.def("upsample_nearest_2d_backward", &nectar::upsample_nearest_2d_backward, 
        py::arg("grad_output_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("H_in"), py::arg("W_in"),
        py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    m.def("upsample_nearest_3d", &nectar::upsample_nearest_3d, 
        py::arg("input_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

    m.def("upsample_nearest_3d", &nectar::upsample_nearest_3d, 
        py::arg("grad_output_ptr"),
        py::arg("B"), py::arg("C"),
        py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
        py::arg("D_out"), py::arg("H_out"), py::arg("W_out"),
        py::arg("dtype"),
        "");

}

