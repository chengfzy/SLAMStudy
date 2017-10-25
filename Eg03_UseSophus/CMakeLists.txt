cmake_minimum_required(VERSION 3.8)
project(Eg03_UseSophus)

set(CMAKE_CXX_STANDARD 11)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)
add_executable(Eg03_UseSophus ${SOURCE_FILES})
target_link_libraries(Eg03_UseSophus ${Sophus_LIBRARIES})