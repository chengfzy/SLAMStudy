cmake_minimum_required(VERSION 3.2)
project(SLAMStudy)

set(CMAKE_CXX_STANDARD 14)

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# libraries
# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${SOPHUS_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
set(G20_LIBS g2o_core g2o_stuff
    g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d g2o_types_slam2d_addons
    g2o_types_slam3d g2o_types_slam3d_addons
    g2o_csparse_extension g2o_solver_cholmod g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_solver_pcg
    g2o_solver_slam2d_linear g2o_solver_structure_only g2o_opengl_helper)

# CSparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# Cholmod
find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})

# DBoW3
find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})

# set extra libraries
set(EXTRA_LIBS ${EXTRA_LIBS} glog gflags ${SOPHUS_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}    
    ${G20_LIBS} ${CSPARSE_LIBRARY} ${CHOLMOD_LIBRARY} ${DBoW3_LIBS})


# sub-module
add_subdirectory(Eg01_UseEigen)
add_subdirectory(Eg02_UseGeometry)
add_subdirectory(Eg03_UseSophus)
add_subdirectory(Eg04_JoinMap)
add_subdirectory(Eg05_UseCeres)
add_subdirectory(Eg06_UseG2O)
add_subdirectory(Eg07_FeatureDetectMatch)
add_subdirectory(Eg08_PoseEstimation2D2D)
add_subdirectory(Eg09_PoseEstimation3D2D)
add_subdirectory(Eg10_PoseEstimation3D3D)

add_subdirectory(Eg15_BagOfWords)
