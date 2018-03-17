cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 14)
project(Eg01_UseEigen)

aux_source_directory(./ DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})
