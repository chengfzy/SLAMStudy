cmake_minimum_required(VERSION 2.8)
project(Eg02_UseGeometry)
set(CMAKE_CXX_STANDARD 17)

# use below commmand, or just find package
# include_directories("/usr/include/eigen3")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)
add_executable(Eg02_UseGeometry ${SOURCE_FILES})
