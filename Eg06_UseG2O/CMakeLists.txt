cmake_minimum_required(VERSION 3.8)
project(Eg06_UseG2O)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(Eg06_UseG2O ${SOURCE_FILES})

target_link_libraries(Eg06_UseG2O ${OpenCV_LIBS} g2o_core g2o_stuff)
