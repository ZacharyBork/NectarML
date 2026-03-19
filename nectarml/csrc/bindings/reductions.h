#include "common.h"

namespace py = pybind11;

namespace nectar {
    uintptr_t reduce_min(uintptr_t in_ptr, size_t n_elements, DType dtype);
    uintptr_t reduce_min_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        DType dtype
    );

    uintptr_t reduce_max(uintptr_t in_ptr, size_t n_elements, DType dtype);
    uintptr_t reduce_max_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        DType dtype
    );

    uintptr_t reduce_mean(uintptr_t in_ptr, size_t n_elements, DType dtype);
    uintptr_t reduce_mean_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        DType dtype
    );

    uintptr_t reduce_sum(uintptr_t in_ptr, size_t n_elements, float initial, DType dtype);
    uintptr_t reduce_sum_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        float initial,
        DType dtype
    );

    uintptr_t reduce_prod(uintptr_t in_ptr, size_t n_elements, float initial, DType dtype);
    uintptr_t reduce_prod_dim(
        uintptr_t in_ptr, 
        std::vector<int> shape,
        int reduce_dim,
        float initial,
        DType dtype
    );
}

void register_reductions(py::module_& m) {

    m.def("reduce_min", &nectar::reduce_min, 
        py::arg("in_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Performs a global min reduction on a CUDA tensor.");

    m.def("reduce_min_dim", &nectar::reduce_min_dim, 
        py::arg("in_ptr"),
        py::arg("shape"),
        py::arg("reduce_dim"),
        py::arg("dtype"),
        "Performs a dimension-wise min reduction on a CUDA tensor.");

    m.def("reduce_max", &nectar::reduce_max, 
        py::arg("in_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Performs a global max reduction on a CUDA tensor.");

    m.def("reduce_max_dim", &nectar::reduce_max_dim, 
        py::arg("in_ptr"),
        py::arg("shape"),
        py::arg("reduce_dim"),
        py::arg("dtype"),
        "Performs a dimension-wise max reduction on a CUDA tensor.");

    m.def("reduce_mean", &nectar::reduce_mean, 
        py::arg("in_ptr"),
        py::arg("n_elements"),
        py::arg("dtype"),
        "Performs a global mean reduction on a CUDA tensor.");

    m.def("reduce_mean_dim", &nectar::reduce_mean_dim, 
        py::arg("in_ptr"),
        py::arg("shape"),
        py::arg("reduce_dim"),
        py::arg("dtype"),
        "Performs a dimension-wise mean reduction on a CUDA tensor.");

    m.def("reduce_sum", &nectar::reduce_sum, 
        py::arg("in_ptr"),
        py::arg("n_elements"),
        py::arg("initial"),
        py::arg("dtype"),
        "Performs a global sum reduction on a CUDA tensor.");

    m.def("reduce_sum_dim", &nectar::reduce_sum_dim, 
        py::arg("in_ptr"),
        py::arg("shape"),
        py::arg("reduce_dim"),
        py::arg("initial"),
        py::arg("dtype"),
        "Performs a dimension-wise sum reduction on a CUDA tensor.");

    m.def("reduce_prod", &nectar::reduce_prod, 
        py::arg("in_ptr"),
        py::arg("n_elements"),
        py::arg("initial"),
        py::arg("dtype"),
        "Performs a global sum reduction on a CUDA tensor.");

    m.def("reduce_prod_dim", &nectar::reduce_prod_dim, 
        py::arg("in_ptr"),
        py::arg("shape"),
        py::arg("reduce_dim"),
        py::arg("initial"),
        py::arg("dtype"),
        "Performs a dimension-wise product reduction on a CUDA tensor.");

}

