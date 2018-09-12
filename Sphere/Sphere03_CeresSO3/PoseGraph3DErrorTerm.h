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
    bool operator()(const T* const p_a, const T* const r_a, const T* const p_b, const T* const r_b, T* residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pA(p_a);
        Eigen::Map<const Sophus::SO3<T>> rA(r_a);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pB(p_b);
        Eigen::Map<const Sophus::SO3<T>> rB(r_b);

        // compute the relative transformation between the two frames
        Eigen::Matrix<T, 3, 1> pAB = rA.inverse() * (pB - pA);
        Sophus::SO3<T> rAB = rA.inverse() * rB;

        // compute the residuals
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residualMat(residual);
        residualMat.template block<3, 1>(0, 0) = pAB - t_ab_.p.template cast<T>();
        residualMat.template block<3, 1>(3, 0) = (rAB.inverse() * t_ab_.R).log();

        // scale the residuals by the measurement uncertainty
        residualMat.applyOnTheLeft(sqrtInfo_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* create(const Pose& t_ab, const Eigen::Matrix<double, 6, 6>& sqrtInformation) {
        return (new ceres::AutoDiffCostFunction<PoseGraph3DErrorTerm, 6, 3, 4, 3, 4>(
            new PoseGraph3DErrorTerm(t_ab, sqrtInformation)));
    }

   private:
    const Pose t_ab_;                             // the measurement for the position of B relative to A in the frame A
    const Eigen::Matrix<double, 6, 6> sqrtInfo_;  // the square root of the measurement information matrix
};