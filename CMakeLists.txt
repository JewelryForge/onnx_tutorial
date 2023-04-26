cmake_minimum_required(VERSION 3.10)
project(onnx_tutorial)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -fopenmp)

include_directories(onnxruntime-linux-x64-1.13.1/include)

add_executable(onnx_tutorial main.cpp onnx_nn.cpp)

target_link_libraries(
    onnx_tutorial ${CMAKE_CURRENT_LIST_DIR}/onnxruntime-linux-x64-1.13.1/lib/libonnxruntime.so
)
