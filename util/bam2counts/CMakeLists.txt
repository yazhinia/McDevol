cmake_minimum_required(VERSION 3.10)
project(bam2counts)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -march=native -Wall -Wextra -Wfloat-conversion --std=c++11 -O3 -fPIC")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -march=native -Wall -Wextra -Wfloat-conversion --std=c++11 -O3 -fPIC")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ..)


add_subdirectory(pybind11)

pybind11_add_module(bam2counts bam2counts.cpp)


target_link_libraries(bam2counts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bamtools/build/src/libbamtools.so")
target_link_libraries(bam2counts PRIVATE -lz)
target_include_directories(bam2counts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bamtools/include/bamtools/")
target_include_directories(bam2counts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bamtools/src/")

