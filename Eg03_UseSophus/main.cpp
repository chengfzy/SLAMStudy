#include <cmath>
#include <iostream>
#include "Eigen/Core"
<<<<<<< HEAD
#include "Eigen/Geometry"
=======
>>>>>>> 013098a54979468654910e65880d9dcaa2dcf0a2
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

using namespace std;

int main() {
    // 沿z轴旋转90度的旋转矩阵
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
<<<<<<< HEAD

=======
>>>>>>> 013098a54979468654910e65880d9dcaa2dcf0a2
    Sophus::SO3d SO3_R(R);    // SO(3)可以直接从旋转矩阵构造
    Eigen::Quaterniond q(R);  // 或者四元数
    Sophus::SO3d SO3_q(q);
    // 输出时，以so(3)形式输出
<<<<<<< HEAD
    cout << "SO(3) from matrix = " << SO3_R.matrix();
    cout << "SO(3) from quaternoion = " << SO3_q.matrix();
=======
    cout << "SO(3) from matrix = " << SO3_R.log().transpose() << endl;
    cout << "SO(3) from quaternoion = " << SO3_q.log().transpose() << endl;
>>>>>>> 013098a54979468654910e65880d9dcaa2dcf0a2

    // 使用对数映射获得它的李代数
    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;
    // hat为向量到反对称矩阵
    cout << "so3 hat = \n" << Sophus::SO3d::hat(so3) << endl;
    // 相对地，vee为反对称到向量
    cout << "so3 hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

    // 增量扰动模型的更新
    Eigen::Vector3d update_so3(1e-4, 0, 0);                            // 假设的更新量
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;  // 左乘更新
    cout << "SO3 updated = " << SO3_updated.matrix() << endl;

    // SE(3)操作大同小异
    cout << "========= SE3 Operation =========" << endl;
    Eigen::Vector3d t(1, 0, 0);  // 沿x轴平移1
    Sophus::SE3d SE3_Rt(R, t);   // 从R,t构造SE(3)
    Sophus::SE3d SE3_qt(q, t);   // 从q,t构造SE(3)
    cout << "SE(3) from R,t = " << endl << SE3_Rt.matrix() << endl;
    cout << "SE(3) from q,t = " << endl << SE3_qt.matrix() << endl;
    // 李代数se(3)是一个六维向量，方便起见先typedef一下
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    // 观察输出，会发现在Sophus中，se(3)平移在前，旋转在后
    cout << "se3 = " << se3.transpose() << endl;
    // 同样，有hat和vee两个算符
    cout << "se3 hat = " << endl << Sophus::SE3d::hat(se3) << endl;
    cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

    // 最后，演示一下更新
    Vector6d update_se3;  // 更新量
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 udpated = " << endl << SE3_updated.matrix() << endl;

    return 0;
}
