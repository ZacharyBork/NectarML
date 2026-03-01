cmake_minimum_required(VERSION 3.18)
find_program(CMAKE_CUDA_COMPILER nvcc
    HINTS
        $ENV{CUDA_HOME}
        $ENV{CUDA_PATH}
        /usr/local/cuda/bin
)
project(nectarml LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find dependencies
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Gather sources
file(GLOB_RECURSE CPP_SOURCES ops/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES kernels/*.cu)

# Build extension module
pybind11_add_module(_nectarml
    bindings.cpp
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Link CUDA runtime
target_link_libraries(_nectarml PRIVATE CUDA::cudart)

# Include csrc/ as root for includes
target_include_directories(_nectarml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CLI override for CUDA architecture:
# pip install . --config-settings=cmake.args="-DCUDA_ARCHITECTURES=70;75;80;86;89;90"
set(CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures to target")
set_target_properties(_nectarml PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

install(TARGETS _nectarml DESTINATION .)
