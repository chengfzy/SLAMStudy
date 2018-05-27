#include <chrono>
#include <iostream>
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;
using namespace g2o;

// Curve Fitting Vertex, <Parameters Dims, Parameters Type>
class CurveFittingVertex : public BaseVertex<3, Eigen::Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // reset
    virtual void setToOriginImpl() override { _estimate << 0, 0, 0; }

    // update
    virtual void oplusImpl(const double* update) override { _estimate += Eigen::Vector3d(update); }

    virtual bool read(istream&) override {}         // load
    virtual bool write(ostream&) const override {}  // save
};

// Residual, <Measurement Dims, Measurement Type, Vertex Type>
class CurveFittingEdge : public BaseUnaryEdge<1, double, CurveFittingVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), x_(x) {}

    // residual
    void computeError() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - exp(abc(0, 0) * x_ * x_ + abc(1, 0) * x_ + abc(2, 0));
    }

    virtual bool read(istream&) override {}         // load
    virtual bool write(ostream&) const override {}  // save

   public:
    double x_;  // data x
};

int main(int argc, char* argv[]) {
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

    // construct graph optimization
    using Block = BlockSolver<g2o::BlockSolverTraits<3, 1>>;  // Block, parameters dims = 3, residual dim = 1
    // linear solver, dense incremental equation
    Block::LinearSolverType* linearSolver = new LinearSolverCSparse<Block::PoseMatrixType>();
    // graph solver, could be Gaussian-Newton, LM, and DogLeg
    OptimizationAlgorithmLevenberg* optAlg = new OptimizationAlgorithmLevenberg(new Block(linearSolver));
    SparseOptimizer optimizer;  // graph model
    optimizer.setAlgorithm(optAlg);
    optimizer.setVerbose(true);  // debug output

    // add vertex to graph
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    // add edge to graph
    for (size_t i = 0; i < N; ++i) {
        CurveFittingEdge* edge = new CurveFittingEdge(xData[i]);
        edge->setId(static_cast<int>(i));
        edge->setVertex(0, v);
        edge->setMeasurement(yData[i]);
        // information matrix: the inverse of covariance matrix
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0 / (wSigma * wSigma));
        optimizer.addEdge(edge);
    }

    // begin to optimize
    auto t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    auto t2 = chrono::steady_clock::now();
    auto timeUsed = chrono::duration_cast<chrono::duration<double, std::milli>>(t2 - t1);  // milli: ms, micro: us
    cout << "solve time used = " << timeUsed.count() << " ms" << endl;

    // print the result
    Eigen::Vector3d absEstimated = v->estimate();
    cout << "estimated = " << absEstimated.transpose() << endl;

    return 0;
}
