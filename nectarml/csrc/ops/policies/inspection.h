#include "common.h"
#include <limits>

/* PREDICATES */

template<typename T>
struct IdentityPred {
    __device__ static bool inspect(volatile T a) { return static_cast<bool>(a); }
};

template<typename T>
struct IsInfPred {
    __device__ static bool inspect(volatile T a) { 
        if constexpr (std::is_same_v<T, half>) { 
            half val = a;
            return static_cast<bool>(__hisinf(val)); 
        } else { return std::isinf(static_cast<float>(a)); }
    }
};

template<typename T>
struct IsFinitePred {
    __device__ static bool inspect(volatile T a) { 
        if constexpr (std::is_same_v<T, half>) { 
            half val = a;
            return !__hisinf(val) && !__hisnan(val);
        } else { return std::isfinite(static_cast<float>(a)); }
    }
};

template<typename T>
struct IsNanPred {
    __device__ static bool inspect(volatile T a) { 
        if constexpr (std::is_same_v<T, half>) { 
            half val = a;
            return static_cast<bool>(__isnan(val)); 
        } else { return std::isnan(static_cast<float>(a)); }
    }
};

/* POLICIES */

struct AllOp {
    __device__ static bool combine(volatile bool& a, volatile bool b) { 
        return a = a && b; }
    __device__ static bool identity() { return true; }
};

struct AnyOp {
    __device__ static bool combine(volatile bool& a, volatile bool b) { 
        return a = a || b; }
    __device__ static bool identity() { return false; }
};

