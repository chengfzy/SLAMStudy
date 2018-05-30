cmake_minimum_required(VERSION 3.2)
project(SLAMStudy)

set(CMAKE_CXX_STANDARD 14)


SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march native")

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# libraries
find_package(Eigen3 REQUIRED)   # Eigen
find_package(Sophus REQUIRED)   # Sophus
find_package(Ceres REQUIRED)    # Ceres
find_package(OpenCV REQUIRED)   # OpenCV
find_package(PCL REQUIRED COMPONENT common io)  # PCL
find_package(CSparse REQUIRED)  # CSparse
find_package(Cholmod REQUIRED)  # Cholmod
find_package(G2O REQUIRED)      # G2O
find_package(DBoW3 REQUIRED)    # DBoW3
include_directories(
    ${EIGEN3_INCLUDE_DIR}       # Eigen
    ${SOPHUS_INCLUDE_DIRS}      # Sophus
    ${CERES_INCLUDE_DIRS}       # Ceres
    ${OpenCV_INCLUDE_DIRS}      # OpenCV
    ${PCL_INCLUDE_DIRS}         # PCL
    ${CSPARSE_INCLUDE_DIR}      # CSparse
    ${CHOLMOD_INCLUDE_DIR}      # Cholmod
    ${G2O_INCLUDE_DIRS}         # G2O
    ${DBoW3_INCLUDE_DIRS}       # DBoW3
)
set(THIRD_LIBS ${THIRD_LIBS} gflags glog ${SOPHUS_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES}
    ${CSPARSE_LIBRARY} ${CHOLMOD_LIBRARY}
    g2o_core g2o_stuff g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d
    g2o_types_slam2d_addons g2o_types_slam3d g2o_types_slam3d_addons g2o_csparse_extension g2o_solver_cholmod
    g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_solver_pcg g2o_solver_slam2d_linear
    g2o_solver_structure_only g2o_opengl_helper
     ${DBoW3_LIBS}
)

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

#add_subdirectory(Eg15_BagOfWords)
