#pragma once

#include "common/dtype.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <cuda_runtime.h>

class CudaMemoryPool {
    std::unordered_map<size_t, std::vector<void*>> pool;
    std::unordered_set<uintptr_t> pool_allocated;
    std::mutex mtx;
    bool enabled     = false;
    bool initialized = false;

    size_t pool_bytes            = 0;
    size_t max_pool_bytes        = 0;
    float  max_pool_vram_percent = 0.2;
    bool   evict_on_oom          = true;

    void initialize_pool();
    static size_t bucket(size_t bytes);
    
public:
    void  enable();
    void  disable(const bool release_pool = true);
    void* alloc(size_t bytes);
    void  free(void* ptr, size_t bytes);
    void  release_unlocked();
    void  release();
};

static CudaMemoryPool g_pool;

