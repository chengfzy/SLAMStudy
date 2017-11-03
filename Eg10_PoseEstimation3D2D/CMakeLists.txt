cmake_minimum_required(VERSION 3.8)
project(Eg10_PoseEstimation3D2D)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(Eg10_PoseEstimation3D2D ${SOURCE_FILES})

target_link_libraries(Eg10_PoseEstimation3D2D ${OpenCV_LIBS})