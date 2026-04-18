#include "common.h"
#include "allocator_pool/allocator_pool.h"
#include <cuda_runtime.h>

size_t CudaMemoryPool::bucket(size_t bytes) {
    size_t p = 1;
    while (p < bytes) p <<= 1;
    return p;
}

void CudaMemoryPool::enable()  { enabled = true; }
void CudaMemoryPool::disable() { enabled = false; }

void* CudaMemoryPool::alloc(size_t bytes) {
    if (!enabled) {
        void* ptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }
    size_t b = bucket(bytes);
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = pool.find(b);
        if (it != pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
    }
    void* ptr;
    cudaMalloc(&ptr, b);
    return ptr;
}

void CudaMemoryPool::free(void* ptr, size_t bytes) {
    if (!enabled || ptr == nullptr) {
        cudaFree(ptr);
        return;
    }
    size_t b = bucket(bytes);
    std::lock_guard<std::mutex> lock(mtx);
    pool[b].push_back(ptr);
}

void CudaMemoryPool::release() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto& [b, ptrs] : pool)
        for (void* ptr : ptrs)
            cudaFree(ptr);
    pool.clear();
}

