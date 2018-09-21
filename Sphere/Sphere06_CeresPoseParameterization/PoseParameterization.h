#pragma once
#include "ceres/local_parameterization.h"
#include "types.h"

class PoseParameterization : public ceres::LocalParameterization {
   public:
    // x: [q, p], delta: [delta phi, delta p]
    // q = q0 * dq; p = p0 + R(q0) * dp
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        Eigen::Map<const Eigen::Quaterniond> q0(x);
        Eigen::Map<const Eigen::Vector3d> p0(x + 4);
        Eigen::Map<const Eigen::Vector3d> dPhi(delta);
        Eigen::Map<const Eigen::Vector3d> dp(delta + 3);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> p(x_plus_delta + 4);

        // convert dPhi to quaternion
        const double normDelta = dPhi.norm();
        // q = q0 * Exp(dPhi), p = p + q0 * dp
        if (normDelta > 0.0) {
            const double sinDeltaByDelta = sin(normDelta / 2.0) / normDelta;
            Eigen::Quaterniond dq(cos(normDelta / 2.0), sinDeltaByDelta * dPhi[0], sinDeltaByDelta * dPhi[1],
                                  sinDeltaByDelta * dPhi[2]);
            q = q0 * dq;
            p = p0 + q0 * dp;
        } else {
            q = q0;
            p = p0 + dp;
        }
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<const Eigen::Quaterniond> q(x);
        Eigen::Map<const Eigen::Vector3d> p(x + 4);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);

        // J = [Dq / DdeltaPhi, 0; 0, Dp / DdeltaP]
        J.setZero();
        double w2 = q.w() * q.w();
        double x2 = q.x() * q.x();
        double y2 = q.y() * q.y();
        double z2 = q.z() * q.z();
        double xy = q.x() * q.y();
        double yz = q.y() * q.z();
        double zx = q.x() * q.z();
        double wx = q.w() * q.x();
        double wy = q.w() * q.y();
        double wz = q.w() * q.z();
        // clang-format off
        J(0, 0) =  0.5 * q.w(); J(0, 1) = -0.5 * q.z(); J(0, 2) =  0.5 * q.y();
        J(1, 0) =  0.5 * q.z(); J(1, 1) =  0.5 * q.w(); J(1, 2) = -0.5 * q.x();
        J(2, 0) = -0.5 * q.y(); J(2, 1) =  0.5 * q.x(); J(2, 2) =  0.5 * q.w();
        J(3, 0) = -0.5 * q.x(); J(3, 1) = -0.5 * q.y(); J(3, 2) = -0.5 * q.z();
        J(4, 3) = w2 + x2 - y2 - z2;    J(4, 4) = 2 * (xy - wz);        J(4, 5) = 2 * (zx + wy);
        J(5, 3) = 2 * (xy + wz);        J(5, 4) = w2 - x2 + y2 - z2;    J(5, 5) = 2 * (yz - wx);
        J(6, 3) = 2 * (zx - wy);        J(6, 4) = 2 * (yz + wx);        J(6, 5) = w2 - x2 - y2 + z2;
        // clang-format on

        return true;
    }

    int GlobalSize() const override { return 7; };

    int LocalSize() const override { return 6; };
};