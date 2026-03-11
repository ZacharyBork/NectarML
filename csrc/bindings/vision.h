#include "common.h"

namespace py = pybind11;

/* TRANSFORMS */

py::array_t<uint8_t> hue_shift(py::array_t<uint8_t> image, float shift);

void register_vision(py::module_& m) {

    /* TRANSFORMS */

    m.def("hue_shift", &hue_shift,
        py::arg("image"),
        py::arg("shift"),
        "Shift the hue of an RGB image by the given number of degrees."
    );
    
}

