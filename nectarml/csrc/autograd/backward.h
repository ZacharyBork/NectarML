#pragma once
#include "graph.h"

inline void _build_topo(
    AutogradNode* node,
    std::vector<AutogradNode*>& topo
) {
    if (!node || node->visited) return;
    node->visited = true;
    for (AutogradNode* input : node->inputs)
        _build_topo(input, topo);
    topo.push_back(node);
}

inline void run_backward(AutogradNode* root) {
    std::vector<AutogradNode*> topo;
    _build_topo(root, topo);
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn)
            (*it)->backward_fn();
        (*it)->backward_fn = nullptr;
        (*it)->inputs.clear();
    }
}