cmake_minimum_required(VERSION 3.8)
project(Eg05_UseCeres)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(Eg05_UseCeres ${SOURCE_FILES})
target_link_libraries(Eg05_UseCeres ${OpenCV_LIBS} ${CERES_LIBRARIES})