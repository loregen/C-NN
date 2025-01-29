cmake_minimum_required(VERSION 3.10)
project(mnistnet C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra -Ofast")
set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g")

# Find BLAS library
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "Using BLAS: ${BLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS not found!")
endif()

# Handle macOS-specific flags
if(APPLE)
    add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
endif()

# Source files
set(SRC_DIR src)
set(SOURCES
    ${SRC_DIR}/net.c
    ${SRC_DIR}/array.c
    ${SRC_DIR}/layers.c
    ${SRC_DIR}/data.c
    ${SRC_DIR}/tensor.c
    ${SRC_DIR}/MNIST_DENSE.c
)

# Define executable for release build
add_executable(mnistnet ${SOURCES})
target_link_libraries(mnistnet m pthread ${BLAS_LIBRARIES})

# Define executable for debug build
add_executable(mnistnet_debug ${SOURCES})
target_compile_options(mnistnet_debug PRIVATE "-g")
target_link_libraries(mnistnet_debug m pthread ${BLAS_LIBRARIES})

# Clean target (handled automatically by CMake's built-in clean)
message(STATUS "Run 'cmake --build . --target clean' to clean binaries.")
