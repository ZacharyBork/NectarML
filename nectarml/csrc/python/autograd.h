#pragma once

#include "autograd/graph.h"
#include "autograd/backward.h"
#include "pool/allocator_pool.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_autograd(py::module_& m) {
    
    auto autograd = m.def_submodule("autograd", "Autograd submodule.");

    m.def("create_node", [](
        py::function           backward_fn,
        std::vector<uintptr_t> input_ptrs
    ) -> uintptr_t {
        std::vector<AutogradNode*> inputs;
        for (uintptr_t ptr : input_ptrs)
            if (ptr) inputs.push_back(reinterpret_cast<AutogradNode*>(ptr));
        
        AutogradNode* node = g_graph.create_node(
            [backward_fn]() { backward_fn(); },
            std::move(inputs));
        
        return reinterpret_cast<uintptr_t>(node);
    });

    m.def("run_backward", [](uintptr_t root_ptr) {
        AutogradNode* root = reinterpret_cast<AutogradNode*>(root_ptr);
        run_backward(root);
    });

    m.def("clear_graph", []() { g_graph.clear(); });
}