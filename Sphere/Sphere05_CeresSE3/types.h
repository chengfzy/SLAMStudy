#pragma once
#include <iostream>
#include <map>
#include <string>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "sophus/se3.hpp"

// The state for each vertex in the pose graph
struct Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sophus::SE3d T;

    // the name of the data type in g2o file format
    static const std::string name() { return "VERTEX_SE3:QUAT"; }
};

// The constraint between two vertices in the pose graph, ie, the transform form vertex id_begin to vertex it_end
struct Constraint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int idBegin;
    int idEnd;

    // the transformation represents the pose of the end frame E w.r.t the begin frame B. In other words, it transforms
    // a vector in the E frame to the B frame
    Pose t_be;

    // the inverse of the covariance matrix for the measurement, the order are x, y, z, delta orientation
    Eigen::Matrix<double, 6, 6> information;

    // the name of the data type in the g2o file format
    static const std::string name() { return "EDGE_SE3:QUAT"; }
};

using MapOfPoses = std::map<int, Pose, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Pose>>>;
using VectorOfConstaints = std::vector<Constraint, Eigen::aligned_allocator<Constraint>>;

// read for Pose
std::istream& operator>>(std::istream& is, Pose& pose) {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    is >> p.x() >> p.y() >> p.z() >> q.x() >> q.y() >> q.z() >> q.w();

    pose.T = Sophus::SE3d(q, p);
    return is;
}

// read for Constraint
std::istream& operator>>(std::istream& is, Constraint& constraint) {
    is >> constraint.idBegin >> constraint.idEnd >> constraint.t_be;

    for (int i = 0; i < 6 && is.good(); ++i) {
        for (int j = i; j < 6 && is.good(); ++j) {
            is >> constraint.information(i, j);
            if (i != j) {
                constraint.information(j, i) = constraint.information(i, j);
            }
        }
    }

    return is;
}