
#include "include/common/dtype.h"
#include "include/common/data_structures.h"
#include <stdint.h>
#include <vector>

enum class Device { CPU, CUDA };

class Tensor {
private:

public:
    Tensor(
        uintptr_t        data_ptr, 
        std::vector<int> shape,
        Device           device,
        DType            dtype,
        bool             requires_grad,
        std::vector<uintptr_t> _children
    );

    uintptr_t         data_ptr;
    std::vector<int>& shape;
    Device            device = Device::CPU;
    DType             dtype  = DType::Float32;
    bool       requires_grad = false;
    std::vector<uintptr_t> _children;

};


