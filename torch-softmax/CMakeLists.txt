cmake_minimum_required(VERSION 3.18)
project(torch_softmax CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake at your LibTorch root when configuring:
#   -DCMAKE_PREFIX_PATH=$HOME/workspace/libtorch
find_package(Torch REQUIRED)

add_executable(torch_softmax torch_softmax.cpp)
target_link_libraries(torch_softmax PRIVATE ${TORCH_LIBRARIES})
target_compile_options(torch_softmax PRIVATE ${TORCH_CXX_FLAGS})

# Make the binary find LibTorch at runtime (Linux)
if(UNIX AND NOT APPLE)
  set_target_properties(torch_softmax PROPERTIES BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib")
endif()