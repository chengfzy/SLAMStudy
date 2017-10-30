#include <iostream>
#include <chrono>
#include "opencv2/core.hpp"
#include "ceres/ceres.h"

using namespace std;


// 代价函数的计算模型
struct CurveFittingCost{
    CurveFittingCost(double x_, double y_) : x(x_), y(y_){}

    //残差的计算
    template <typename T>
    bool operator()(const T* const abc, T* residual) const{
        //y - exp(ax^2 + bx + c)
        residual[0] = T(y) - ceres::exp(abc[0] * T(x) * T(x) + abc[1] * T(x) + abc[2]);
        return true;
    }

    const double  x, y;     // x, y数据
};

int main() {
    double a = 1.0, b = 2.0, c = 1.0;       // 真实参数值
    int N = 100;                            // 数据点
    double wSigma = 1.0;                   // 噪声sigma值
    cv::RNG rng;                            // 随机数产生器
    double abc[3] = {0, 0, 0};              // abc参数的估计值

    vector<double> xData, yData;            // 数据

    cout << "generating data..." << endl;
    for (int i = 0; i < N; ++i){
        double x = i / 100.0;
        xData.emplace_back(x);
        yData.emplace_back(exp(a * x * x + b * x + c) + rng.gaussian(wSigma));
        cout << xData[i] << ", " << yData[i] << endl;
    }

    // 构建最小二乘问题
    ceres::Problem problem;
    for (int i = 0; i < N; ++i){
        problem.AddResidualBlock(
                // 向问题中添加误差项，使用自动求导，模板参数：误差类型，输出维度、输入维度，数值参数参照前面struct中写法
                new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(
                        new CurveFittingCost(xData[i], yData[i])
                ),
                nullptr,    // 核函数，这里不使用为空
                abc         // 待估计参数
        );
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;    // 输出到cout

    ceres::Solver::Summary summary;                 // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);      // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeUsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << timeUsed.count() << " seconds" << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a, b, c = ";
    for (auto a: abc){
        cout << a << " ";
    }
    cout << endl;

    return 0;
}