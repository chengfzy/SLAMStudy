# SLAM
Some Study Codes and Projects about SLAM

# Contents
1. Eg01_UseEigen    \
    Use Eigen library
1. Eg02_UseGeometry \
    Some conversions between different geometry representation, like rotation matrix, angle axis, euler angles and quaterniond
1. Eg03_UseSophus   \
    use Sophus library to study Lie algebra and groups
1. Eg04_JoinMap \
    Use PCL(Point Cloud Library) to join map from RGB-D images
1. Eg05_UseCeres    \
    Curve fit using Ceres
1. Eg06_UseG2O  \
    Curve fit using g2o
1. Eg07_FeatureDetectMatch  \
    Feature detect and match using OpenCV, and use features to estimate pose in 2D-2D
1. Eg08_PoseEstimation2D2D  \
    Estimate camera pos 2D-2D using epipolar geometry costraints, and triangulatio to obtain 3D point in world.
1. Eg09_PoseEstimation3D2D  \
    Get camera pose using PnP algorithm, and use bundle adjustment to optimize the pose.
1. Eg10_PoseEstimation3D3D \
    Get camera pose using ICP algorithm(SVD and bundle adjustment) based on 3D-3D point pairs.
1. Eg11_Triangulation   \
    Triangulation to obtain position of map point
1. Eg15_BagOfWords \
    Generate vocabulary from images and calculate the simularity among images, dataset comes from
    [TUM RGBD dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#). Note the result using large
    vocabulary file don't seem good enough as books.

# G2O Examples
1. g2o01_CurveFit   \
    Using g2o for curve fitting.
    Ref: https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/data_fitting/curve_fit.cpp.
1. g2o02_CircleFit  \
    Using g2o for circle fitting, and compare with another solution, construct a linear least square problem solved by Eigen.
    Ref: [circle_fit.cpp](https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/data_fitting/circle_fit.cpp)

# Sphere Optimization
Read the data from file sphere2500.g20, and then optimize it with different method.
1. Sphere01_CeresQuaternion \
    Optimize it using ceres and rotation is represented by quaternion, with AutoDiffCostFunction.
    Ref: https://github.com/ceres-solver/ceres-solver/tree/master/examples/slam/pose_graph_3d
1. Sphere02_CeresQuaternion \
    Same as Sphere01_CeresQuaternion, but the parameter block are pose(quaternion and position), and use 
    ProductParameterization for local parameterization.
1. Sphere03_CeresSO3    \
    Optimize the problem useing ceres, and rotation is represensted by SO3 with SO3Parameterization.
1. Sphere04_CeresSO3    \
    Same as Sphere03_CeresSO3, but the parameter block are pose(SO3 and position), and use ProductParameterization 
    for local parameterization.
1. Sphere05_CeresSE3    \
    Optimize it using ceres, pose(rotation and translation) is represented by SE3 with SE3Parameterization.
1. Sphere06_CeresPoseParameterization   \
    Same as Sphere02_CeresQuaternion, but the parameter block are pose(quater and position), and use defined
    parameterization method, $q = q0 \times Exp(\delta phi), p = p0 + q0 \times \delta p \times \q0^*$. But
    the cost function are calculated by AutoDiff using Jet, and found that the jacobians calculation may be
    some error and the result will not convergency. If change the cost function to SizedFunction maybe solve
    this problem.

# Problems
1. Some error occurs about LinearSolverCholmod/LinearSolverCSparse, and cannot find solution, try again later.

    Change FindCXSparse.cmake to shared library, Eg06 will be OK but others not.

# Note
1. This repository will not be maintained right now, some of the example will be move to [CPlusPlusStudy](https://github.com/chengfzy/CPlusPlusStudy) and [VisionStudy](https://github.com/chengfzy/VisionStudy) with more better project stucture.
