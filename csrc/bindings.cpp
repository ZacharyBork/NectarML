#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<uint8_t> hue_shift(py::array_t<uint8_t> image, float shift);

PYBIND11_MODULE(_nectarml, m) {
    m.doc() = "NectarML C++ extension module";

    m.def("hue_shift", &hue_shift,
        py::arg("image"),
        py::arg("shift"),
        "Shift the hue of an RGB image by the given number of degrees."
    );
}
