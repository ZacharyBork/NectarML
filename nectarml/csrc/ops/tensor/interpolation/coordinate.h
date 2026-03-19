#include "common.h"

template<bool align_corners>
struct AlignCorners;

template<>
struct AlignCorners<true> {
    __device__ static float operation(int coord_in, int coord_out) {
        return (float)(coord_in - 1) / (coord_out - 1);
    }
};

template<>
struct AlignCorners<false> {
    __device__ static float operation(int coord_in, int coord_out) {
        return (float)coord_in / coord_out;
    }
};

