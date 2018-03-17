cmake_minimum_required(VERSION 3.2)
project(Eg10_PoseEstimation3D2D)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${G2O_INCLUDE_DIRS} ${CSPARSE_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)
add_executable(Eg10_PoseEstimation3D2D ${SOURCE_FILES})

target_link_libraries(Eg10_PoseEstimation3D2D ${OpenCV_LIBS} ${CSPARSE_LIBRARY}
        g2o_core g2o_stuff g2o_types_sba g2o_types_sim3 g2o_csparse_extension)
