#pragma once

/* OPERATION CONSTANTS */

#define MAX_DIMS          6  // Maximum allowable number of tensor dimensions.
#define MAX_CONCAT_INPUTS 32 // Max input count for tensor combination ops
#define MAX_PERMUTE_DIMS  6  // Max dims for tensor permutation ops.

/* ALLOCATION CONSTANTS */

constexpr int BLOCK_SIZE_1D = 256; // Block size constant for 1D allocation
constexpr int BLOCK_SIZE_2D = 16;  // Block size constant for 2D allocation


