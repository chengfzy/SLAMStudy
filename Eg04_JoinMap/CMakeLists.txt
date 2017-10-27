cmake_minimum_required(VERSION 3.2)
project(Eg04_JoinMap)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENT common io)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(Eg04_JoinMap ${SOURCE_FILES})
target_link_libraries(Eg04_JoinMap ${OpenCV_LIBS} ${PCL_LIBRARIES})