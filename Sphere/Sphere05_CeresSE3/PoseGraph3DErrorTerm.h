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
        Eigen::Map<const Sophus::SE3<T>> pA(p_a);
        Eigen::Map<const Sophus::SE3<T>> pB(p_b);

        Sophus::SE3<T> pAB = pA.inverse() * pB;
        Sophus::SE3<T> res = pAB.inverse() * t_ab_.T.template cast<T>();

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residualMat(residual);
        residualMat = res.log();
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