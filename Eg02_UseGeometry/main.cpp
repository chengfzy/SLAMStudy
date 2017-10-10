#include <iostream>
#include <cmath>
#include "Eigen/Core"
#include "Eigen/Geometry"

using namespace std;

int main() {
    // 3D旋转矩阵直接用Matrix3d或Matrix3f
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    // 旋转微量用AngleAxis
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));  // 沿z轴旋转45度
    cout.precision(3);
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;     // 用matrix()转换成矩阵
    rotation_matrix = rotation_vector.toRotationMatrix();
    // 用AngleAxis进行坐标变换
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    // 或者用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    // 欧拉角,可以将旋转矩阵直接转换成角
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);    // ZYX顺序,即yaw, pitch, roll
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    // 欧氏变换矩阵使用Eigen::Isometry
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();    // 虽然称为3d,实质上是4X4矩阵
    T.rotate(rotation_vector);      // 按照rotation_vector进行旋转
    T.pretranslate(Eigen::Vector3d(1, 3, 4));   // 将向量平移设成(1,3,4)
    cout << "Transform matrix = \n" << T.matrix() << endl;

    // 用变换矩阵进行坐标变换
    Eigen::Vector3d v_transformed = T * v;      // 相当于R*v+t
    cout << "v transformed = " << v_transformed.transpose() << endl;

    // 对于仿射和射影变换,使用Eigen::Affine3d和Eigen::Projective3d即可,略

    // 四元数
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector); // 直接把AngleAxis赋值给四元数
    cout << "quaternion = \n" << q.coeffs() << endl;    // 注意coeffs的顺序是(x,y,z,w), w为实部
    q = Eigen::Quaterniond(rotation_matrix);        // 也可以将旋转矩阵赋值给他
    cout << "quaternion = \n" << q.coeffs() << endl;
    // 使用四元数旋转一个向量,使用重载的乘法即可,注意数学上是qvq^{-1}
    v_rotated = q * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    return 0;
}