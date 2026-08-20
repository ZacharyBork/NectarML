#include "pool/allocator_pool.h"

void CudaMemoryPool::initialize_pool() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    max_pool_bytes = (size_t)((float)total * max_pool_vram_percent);
    _initialized    = true;
}

size_t CudaMemoryPool::bucket(size_t bytes) {
    size_t p = 1;
    while (p < bytes) p <<= 1;
    return p;
}

void CudaMemoryPool::enable()  { 
    if (!_initialized) initialize_pool();
    _enabled = true;
}
void CudaMemoryPool::disable(const bool release_pool) { 
    _enabled = false;
    if (release_pool) release();
}

void CudaMemoryPool::set_vram_percent(float percent) {
    max_pool_vram_percent = percent;
    initialize_pool();
}

void CudaMemoryPool::set_evict_on_oom(const bool enabled) {
    evict_on_oom = enabled;
}

void* CudaMemoryPool::alloc(size_t bytes) {
    if (!_initialized) initialize_pool();
    if (!_enabled) {
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
            pool_bytes -= b;
            return ptr;
        }
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, b);
    
    if (err == cudaErrorMemoryAllocation) {
        if (evict_on_oom) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                release_unlocked();
            }
            err = cudaMalloc(&ptr, b);
            if (err != cudaSuccess)
                throw std::runtime_error("CUDA out of memory after pool release");
        }  else throw std::runtime_error("CUDA out of memory");
    }
    
    {
        std::lock_guard<std::mutex> lock(mtx);
        pool_allocated.insert(reinterpret_cast<uintptr_t>(ptr));
    }
    
    return ptr;
}

void CudaMemoryPool::free(void* ptr, size_t bytes) {
    if (!ptr) return;
    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    
    bool should_hard_free = false;
    size_t b = bucket(bytes);
    
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (pool_allocated.count(uptr) == 0) {
            should_hard_free = true;
        } else if (!_enabled || pool_bytes + b > max_pool_bytes) {
            pool_allocated.erase(uptr);
            should_hard_free = true;
        } else {
            pool_bytes += b;
            pool[b].push_back(ptr);
        }
    }
    
    if (should_hard_free) cudaFree(ptr);
}

void CudaMemoryPool::release_unlocked() {
    for (auto& [b, ptrs] : pool)
        for (void* ptr : ptrs)
            cudaFree(ptr);
    pool.clear();
    pool_allocated.clear();
    pool_bytes = 0;
}

void CudaMemoryPool::release() {
    std::lock_guard<std::mutex> lock(mtx);
    release_unlocked();
}



