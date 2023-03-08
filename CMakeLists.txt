cmake_minimum_required(VERSION 3.2)

project(kmerfreq)

set(CMAKE_CXX_STANDARD 17)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

add_executable(kmerfreq kmerfreq.cpp)

include_directories("/big/work/metadevol/scripts/")