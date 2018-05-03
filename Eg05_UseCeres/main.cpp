#include <chrono>
#include <iostream>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"

using namespace std;

// cost function
struct CurveFittingCost {
    CurveFittingCost(double x, double y) : x_(x), y_(y) {}

    // residual calculation, y - exp(a*x^2 + b*x + c)
    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] = static_cast<T>(y_) - ceres::exp(abc[0] * static_cast<T>(x_) * static_cast<T>(x_) +
                                                      abc[1] * static_cast<T>(x_) + abc[2]);
        return true;
    }

    const double x_;  // data x
    const double y_;  // data y
};

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    double a{1.0}, b{2.0}, c{1.0};  // true value
    double wSigma{1.0};             // noise sigma
    size_t N{100};                  // data length(size)

    // generating data
    cout << "generating data..." << endl;
    vector<double> xData, yData;
    cv::RNG rng;  // random generator
    for (size_t i = 0; i < N; ++i) {
        double x = i / 100.0;
        xData.emplace_back(x);
        yData.emplace_back(exp(a * x * x + b * x + c) + rng.gaussian(wSigma));
        cout << "[" << i << "] " << xData[i] << ", " << yData[i] << endl;
    }

    // construct least square problem
    ceres::Problem problem;
    double abc[3] = {0, 0, 0};  // initial value
    for (size_t i = 0; i < N; ++i) {
        problem.AddResidualBlock(
            // Auto diff, <CostFunction, Output(Residual) Dimension, Input(Parameters) Dimension>
            new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(new CurveFittingCost(xData[i], yData[i])),
            nullptr,  // loss function(kernel function), nullptr imply squared norm of residuals
            abc       // parameters
        );
    }

    // solver config
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  // incremental function solver
    options.minimizer_progress_to_stdout = true;   // print to console

    // solve
    ceres::Solver::Summary summary;  // optimization info
    auto t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    auto t2 = chrono::steady_clock::now();
    auto timeUsed = chrono::duration_cast<chrono::duration<double, std::milli>>(t2 - t1);  // milli: ms, micro: us
    cout << "solve time used = " << timeUsed.count() << " ms" << endl;

    // print the result
    cout << summary.BriefReport() << endl;
    cout << "estimated a, b, c = " << abc[0] << ", " << abc[1] << ", " << abc[2] << endl;

    return 0;
}
