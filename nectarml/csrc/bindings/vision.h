#pragma once

#include "common/dtype.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

/* TRANSFORMS */

namespace nectar {
    
    /* COLOR */

    uintptr_t hsv_adjust(
        uintptr_t in_ptr, std::vector<int> shape, 
        float hue_shift, float saturation, float value,
        DType dtype
    );

    /* SPATIAL */

    uintptr_t rotate(
        uintptr_t in_ptr, std::vector<int> shape, 
        float angle, float fill_value,
        DType dtype
    );

    /* LUT */

    uintptr_t apply_lut(
        uintptr_t input_ptr,
        uintptr_t lut_ptr,
        int B, int H, int W,
        int lut_size,
        DType dtype
    );

}

void register_vision(py::module_& m) {

    /* COLOR */

    m.def("hsv_adjust", &nectar::hsv_adjust,
        py::arg("in_ptr"), py::arg("shape"),
        py::arg("hue_shift"), py::arg("saturation"), py::arg("value"),
        py::arg("dtype"),
        "");

    /* SPATIAL */
    
    m.def("rotate", &nectar::rotate,
        py::arg("in_ptr"), py::arg("shape"),
        py::arg("angle"), py::arg("fill_value"),
        py::arg("dtype"),
        "");

    /* LUT */
    
    m.def("apply_lut", &nectar::apply_lut,
        py::arg("in_ptr"), py::arg("lut_ptr"),
        py::arg("B"), py::arg("H"), py::arg("W"),
        py::arg("lut_size"), 
        py::arg("dtype"),
        "");
}

