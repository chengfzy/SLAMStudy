cmake_minimum_required(VERSION 3.2)
project(Eg09_Triangulation)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(Eg09_Triangulation ${SOURCE_FILES})

target_link_libraries(Eg09_Triangulation ${OpenCV_LIBS})
