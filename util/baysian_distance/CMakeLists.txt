cmake_minimum_required(VERSION 3.10)
project(metadevol_distance)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -march=native -Wall -Wextra -Wfloat-conversion -Wsign-compare")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -march=native -Wall -Wextra -Wfloat-conversion -Wsign-compare")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ..)

add_subdirectory(pybind11)

find_package(OpenMP REQUIRED)

pybind11_add_module(metadevol_distance pythonBinding.cpp)
target_link_libraries(metadevol_distance PRIVATE OpenMP::OpenMP_CXX)
