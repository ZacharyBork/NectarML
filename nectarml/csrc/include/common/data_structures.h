#pragma once

#include "include/constants.h"
#include <vector>

/* TENSOR INDEX */

struct TensorIndex {
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
    int n_elements;

    __host__ __device__ TensorIndex() : ndim(0), n_elements((0)) {
        for(int i = 0; i < MAX_DIMS; i++) {
            shape[i] = 0;
            strides[i] = 0;
        }
    }

    __host__ __device__ TensorIndex(
        const int* shape_, 
        int ndim_
    ) : ndim(ndim_) {
        n_elements = 1;
        for (int i = 0; i < ndim; i++) {
            shape[i] = shape_[i];
            n_elements *= shape[i];
        }
        for (int i = 0; i < MAX_DIMS - ndim; i++) 
            shape[ndim + i] = 0;
        _compute_strides();
    }

    __host__ __device__ void _compute_strides() {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
    }

    __host__ __device__ int to_flat(int* indices) {
        int flat = 0;
        for(int i = 0; i < ndim; i++) 
            flat += indices[i] * strides[i];
        return flat;
    }

    __host__ __device__ void to_index(int flat, int* out_indices) {
        for(int i = ndim-1; i >= 0; i--) {
            out_indices[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
};

inline TensorIndex build_tensor_index(const std::vector<int>& shape) {
    return TensorIndex(shape.data(), shape.size());
}

/* COMBINATION */

struct ConcatInputs {
    uintptr_t ptrs[MAX_CONCAT_INPUTS];
    TensorIndex indices[MAX_CONCAT_INPUTS];
    int offsets[MAX_CONCAT_INPUTS];
    int n_inputs;
};

/* BROADCASTING */

struct BroadcastIndex {
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
    
    __device__ int get_flat(int coords[]) const {
        int flat = 0;
        for (int i = 0; i < ndim; i++) { flat += coords[i] * strides[i]; }
        return flat;
    }
};

struct ShapeArray {
    int dims[MAX_DIMS];
    int ndim;
};

/* SLICING */

struct SliceIndex {
    int start[MAX_DIMS];
    int step[MAX_DIMS];
    int ndim;
    
    SliceIndex(int* start, int* step, int ndim) {
        for (int i = 0; i < ndim; i++) {
            this->start[i] = start[i];
            this->step[i]  = step[i];
        }
        this->ndim = ndim;
    }
};

/* PERMUTATION */

struct Permutation {
    int dims[MAX_PERMUTE_DIMS];
    int ndim;
    
    __host__ __device__ Permutation() : ndim(0) {
        for (int i = 0; i < MAX_PERMUTE_DIMS; i++) dims[i] = 0;
    }
    
    __host__ __device__ Permutation(const int* dims_, int ndim_) : ndim(ndim_) {
        for (int i = 0; i < ndim; i++) dims[i] = dims_[i];
    }
    
    __host__ __device__ Permutation inverse() const {
        Permutation inv;
        inv.ndim = ndim;
        for (int i = 0; i < ndim; i++)
            inv.dims[dims[i]] = i;
        return inv;
    }
};

