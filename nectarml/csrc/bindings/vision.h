#include "common.h"

namespace py = pybind11;

/* TRANSFORMS */

namespace nectar {
    uintptr_t hsv_adjust(
        uintptr_t in_ptr, std::vector<int> shape, 
        float hue_shift, float saturation, float value,
        DType dtype
    );
}

void register_vision(py::module_& m) {

    /* TRANSFORMS */

    m.def("hsv_adjust", &nectar::hsv_adjust,
        py::arg("in_ptr"), py::arg("shape"),
        py::arg("hue_shift"), py::arg("saturation"), py::arg("value"),
        py::arg("dtype"),
        "");
    
}

