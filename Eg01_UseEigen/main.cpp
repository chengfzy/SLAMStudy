#include <ctime>
#include <iostream>

#include <Eigen/Core>
// 稠密矩阵的代数运算(逆、特征值等)
#include <Eigen/Dense>

using namespace std;

#define MATRIX_SIZE 50

int main() {
    // define
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;

    // output
    cout << "output ..." << endl;
    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << matrix_23 << endl;

    // access
    cout << "access ..." << endl;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << matrix_23(i, j) << endl;
        }
    }

    // vector * matrix
    cout << "vector * matrix ..." << endl;
    v_3d << 3, 2, 1;
    // 不能混用不同类型的矩阵，应该转换，并且维度要正确
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << result << endl;

    // matrix operation ...
    cout << "matrix operation ..." << endl;
    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl;
    cout << "transponse: \n" << matrix_33.transpose() << endl;
    cout << "sum: \n" << matrix_33.sum() << endl;
    cout << "trace: \n" << matrix_33.trace() << endl;
    cout << "times: \n" << 10 * matrix_33 << endl;
    cout << "inverse: \n" << matrix_33.inverse() << endl;
    cout << "determinant: \n" << matrix_33.determinant() << endl;

    // eigen value
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(matrix_33.transpose() * matrix_33);
    cout << "eigen value: \n" << eigenSolver.eigenvalues() << endl;
    cout << "eigen vectors: \n" << eigenSolver.eigenvectors() << endl;

    // solve: matrix_NN * x = v_Nd：直接求逆与矩形分解计算
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_start = clock();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time use in normal inverse is " << 1000 * (clock() - time_start) / static_cast<double>(CLOCKS_PER_SEC)
         << " ms" << endl;

    time_start = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time use in QR composition is " << 1000 * (clock() - time_start) / static_cast<double>(CLOCKS_PER_SEC)
         << " ms" << endl;

    return 0;
}
