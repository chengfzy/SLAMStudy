#pragma once
#include "ceres/local_parameterization.h"
#include "sophus/se3.hpp"

class SE3Parameterization : public ceres::LocalParameterization {
   public:
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        Eigen::Map<const Sophus::SE3d> T0(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> dPhi(delta);
        Eigen::Map<Sophus::SE3d> T1(x_plus_delta);

        T1 = T0 * Sophus::SE3d::exp(dPhi);
        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        Eigen::Map<const Sophus::SE3d> T(x);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobianMat(jacobian);

        jacobianMat = T.Dx_this_mul_exp_x_at_0();
        return true;
    }

    // Size of x: 7
    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

    // Size of delta: 6
    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};