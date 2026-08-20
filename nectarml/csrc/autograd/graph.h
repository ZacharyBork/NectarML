#pragma once

#include <functional>
#include <vector>
#include <memory>
#include <unordered_map>

struct AutogradNode {
    std::function<void()>      backward_fn;
    std::vector<AutogradNode*> inputs;
    bool                       visited = false;
};

struct AutogradGraph {
    std::vector<std::unique_ptr<AutogradNode>> nodes;
    
    AutogradNode* create_node(
        std::function<void()>     backward_fn,
        std::vector<AutogradNode*> inputs
    ) {
        auto node = std::make_unique<AutogradNode>();
        node->backward_fn = std::move(backward_fn);
        node->inputs      = std::move(inputs);
        AutogradNode* raw = node.get();
        nodes.push_back(std::move(node));
        return raw;
    }
    
    void clear() { nodes.clear(); }
};

inline thread_local AutogradGraph g_graph;
