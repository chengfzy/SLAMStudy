cmake_minimum_required(VERSION 2.8)
project(Eg02_UseGeometry)

set(CMAKE_CXX_STANDARD 17)

include_directories("/usr/include/eigen3")

set(SOURCE_FILES main.cpp)
add_executable(Eg02_UseGeometry ${SOURCE_FILES})