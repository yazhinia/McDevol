cmake_minimum_required(VERSION 3.2)

project(select_contigsby_length)

set(CMAKE_CXX_STANDARD 17)

add_executable(select_contigsby_length select_contigsby_length.cpp)

target_include_directories(bamtools PRIVATE "/usr/include/bamtools/")
target_include_directories(bamtools PRIVATE "/usr/bin/")
target_link_libraries(bamtools "/usr/lib/x86_64-linux-gnu/libbamtools.a")
target_link_libraries(bamtools -lz)
