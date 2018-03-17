cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 14)
project(Eg02_UseGeometry)

# build the target
aux_source_directory(./ DIR_SRCS)
add_executable(Eg02_UseGeometry ${DIR_SRCS})
