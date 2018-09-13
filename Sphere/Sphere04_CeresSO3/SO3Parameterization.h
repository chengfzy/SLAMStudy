#pragma once
#include "ceres/local_parameterization.h"
#include "sophus/so3.hpp"

class SO3Parameterization : public ceres::LocalParameterization {
   public:
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        Eigen::Map<const Sophus::SO3d> R0(x);
        Eigen::Map<const Eigen::Matrix<double, 3, 1>> dPhi(delta);
        Eigen::Map<Sophus::SO3d> R1(x_plus_delta);

        R1 = R0 * Sophus::SO3d::exp(dPhi);
        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        Eigen::Map<const Sophus::SO3d> T(x);
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jacobianMat(jacobian);

        jacobianMat = T.Dx_this_mul_exp_x_at_0();
        return true;
    }

    // Size of x: 4
    virtual int GlobalSize() const { return Sophus::SO3d::num_parameters; }

    // Size of delta: 3
    virtual int LocalSize() const { return Sophus::SO3d::DoF; }
};