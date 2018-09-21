#pragma once
#include "Eigen/Core"
#include "ceres/ceres.h"
#include "types.h"

class PoseGraph3DErrorTerm {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseGraph3DErrorTerm(const Pose& t_ab, const Eigen::Matrix<double, 6, 6>& sqrtInformation)
        : t_ab_(t_ab), sqrtInfo_(sqrtInformation) {}

    template <typename T>
    bool operator()(const T* const p_a, const T* const p_b, T* residual) const {
        Eigen::Map<const Eigen::Quaternion<T>> qA(p_a);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pA(p_a + 4);
        Eigen::Map<const Eigen::Quaternion<T>> qB(p_b);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pB(p_b + 4);

        // compute the relative transformation between the two frames
        Eigen::Matrix<T, 3, 1> pAB = qA.inverse() * (pB - pA);
        Eigen::Quaternion<T> qAB = qA.inverse() * qB;

        // compute the residuals
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residualMat(residual);
        residualMat.template block<3, 1>(0, 0) = pAB - t_ab_.p.template cast<T>();

        // r_phi = Log(dq)
        Eigen::Quaternion<T> dq = qAB.inverse() * (t_ab_.q.template cast<T>());
        T norm2Dqv = dq.vec().squaredNorm();
        // use the below version to calculate Log(dq) will no be convergence for this problem, maybe caused by jacobian
        // calculation using jet
        // if (norm2Dqv < 1e-6) {
        // approximate: r_phi ~ 2 * [dq_x, dq_y. dq_z]
        // residualMat.template block<3, 1>(3, 0) = T(2.0) * copysign(1, dq.w()) * dq.vec();
        residualMat.template block<3, 1>(3, 0) = T(2.0) * dq.vec();
        /*} else {
            // r_phi = Log(dq)
            T normDqv = sqrt(norm2Dqv);
            residualMat.template block<3, 1>(3, 0) = T(2.0) * atan2(normDqv, dq.w()) * dq.vec() / normDqv;
        }*/

        // scale the residuals by the measurement uncertainty
        residualMat.applyOnTheLeft(sqrtInfo_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* create(const Pose& t_ab, const Eigen::Matrix<double, 6, 6>& sqrtInformation) {
        return (new ceres::AutoDiffCostFunction<PoseGraph3DErrorTerm, 6, 7, 7>(
            new PoseGraph3DErrorTerm(t_ab, sqrtInformation)));
    }

   private:
    const Pose t_ab_;                             // the measurement for the position of B relative to A in the frame A
    const Eigen::Matrix<double, 6, 6> sqrtInfo_;  // the square root of the measurement information matrix
};