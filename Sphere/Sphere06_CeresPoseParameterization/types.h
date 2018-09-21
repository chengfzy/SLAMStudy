#pragma once
#include <iostream>
#include <map>
#include <string>
#include "Eigen/Core"
#include "Eigen/Geometry"

// the state for each vertex in the pose graph
struct Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    inline const double* data() const { return q.coeffs().data(); }
    double* data() { return q.coeffs().data(); }

    Eigen::Quaterniond q;
    Eigen::Vector3d p;

    // the name of the data type in g2o file format
    static const std::string name() { return "VERTEX_SE3:QUAT"; }
};

// the constraint between two vertices in the pose graph, ie, the transform form vertex id_begin to vertex it_end
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

// read for Pose2d
std::istream& operator>>(std::istream& is, Pose& pose) {
    is >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y() >> pose.q.z() >> pose.q.w();

    // normalize the quaternion to account for precision loss due to serialization
    pose.q.normalize();
    return is;
}

// read for Constraint2d
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