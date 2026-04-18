#pragma once

#include "allocator_pool/allocator_pool.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_allocator_pool(py::module_& m) {
    
    auto alloc_pool = m.def_submodule(
        "alloc_pool", "Allocator pool submodule.");

    alloc_pool.def("pool_enable",  []() { g_pool.enable();  });
    alloc_pool.def("pool_disable", []() { g_pool.disable(); });
    alloc_pool.def("pool_release", []() { g_pool.release(); });

}

