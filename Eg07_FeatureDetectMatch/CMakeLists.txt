cmake_minimum_required(VERSION 3.2)
project(Eg07_FeatureDetectMatch)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(Eg07_FeatureDetectMatch ${SOURCE_FILES})

target_link_libraries(Eg07_FeatureDetectMatch ${OpenCV_LIBS})
