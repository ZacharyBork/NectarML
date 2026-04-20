#pragma once

#include "common/dtype.h"
#include "allocator_pool/allocator_pool.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_allocator_pool(py::module_& m) {
    
    auto allocator_pool = m.def_submodule(
        "allocator_pool", "Allocator pool submodule.");

    allocator_pool.def("enable",  []() { g_pool.enable();  });
    allocator_pool.def(
        "disable", [](const bool release_pool = true) { 
            g_pool.disable(release_pool); 
        });
    allocator_pool.def("release", []() { g_pool.release(); });

}

