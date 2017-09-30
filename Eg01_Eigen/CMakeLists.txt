cmake_minimum_required(VERSION 2.8)
project(Eg01_Eigen)

set(CMAKE_CXX_STANDARD 17)

include_directories("/usr/include/eigen3")

set(SOURCE_FILES main.cpp)
add_executable(Eg01_Eigen ${SOURCE_FILES})