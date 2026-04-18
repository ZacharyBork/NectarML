#pragma once

#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace py = pybind11;

class CudaMemoryPool {
    std::unordered_map<size_t, std::vector<void*>> pool;
    std::mutex mtx;
    bool enabled = false;

    static size_t bucket(size_t bytes);

public:
    void enable();
    void disable();
    void* alloc(size_t bytes);
    void free(void* ptr, size_t bytes);
    void release();
};

static CudaMemoryPool g_pool;

