/*
 * Use g2o for curve fitting: y(t) = a * exp(-lambda * t) + b
 * The parameters is a, b and lambda
 */

#include <chrono>
#include <iostream>
#include "Common.hpp"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/stuff/sampler.h"

using namespace std;
using namespace g2o;

// Curve Fitting Vertex, the parameter of a, b and c. <Parameters Dims, Parameters Type>
class CurveFittingVertex : public BaseVertex<3, Eigen::Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // reset
    void setToOriginImpl() override { _estimate << 0, 0, 0; }

    // update
    void oplusImpl(const double* update) override { _estimate += Eigen::Map<const Eigen::Vector3d>(update); }

    // read
    bool read(istream&) override {
        cout << __FUNCTION__ << "not implemented yet" << endl;
        return false;
    }

    // write
    bool write(ostream&) const override {
        cout << __PRETTY_FUNCTION__ << "not implemented yet" << endl;
        return false;
    }
};

// Residual, <Measurement Dims, Measurement Type, Vertex Type>
// Measurement: (x, y)
class CurveFittingEdge : public BaseUnaryEdge<1, Eigen::Vector2d, CurveFittingVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // residual
    void computeError() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const double& a = v->estimate()(0);
        const double& b = v->estimate()(1);
        const double& lambda = v->estimate()(2);
        double fval = a * exp(-lambda * measurement()(0)) + b;
        _error(0) = measurement()(1) - fval;
    }

    bool read(istream&) override {
        cout << __PRETTY_FUNCTION__ << "not implemented yet" << endl;
        return false;
    }

    bool write(ostream&) const override {
        cout << __PRETTY_FUNCTION__ << "not implemented yet" << endl;
        return false;
    }
};

int main(int argc, char* argv[]) {
    double a{2.0}, b{0.4}, lambda{0.2};  // true value
    double wSigma{0.02};                 // noise sigma
    size_t N{100};                       // data length(size)

    // generating data, y = a * exp(-lambda * x) + b + w, w is the Gaussian noise
    cout << section("generating data") << endl;
    vector<Eigen::Vector2d> points(N);
    for (size_t i = 0; i < N; ++i) {
        double x = g2o::Sampler::uniformRand(0, 10);
        ;
        double y = a * exp(-lambda * x) + b + g2o::Sampler::gaussRand(0, wSigma);
        points[i].x() = x;
        points[i].y() = y;
        cout << "[" << i << "] " << points[i].x() << ", " << points[i].y() << endl;
    }

    // some type definitions
    using BlockSolver = BlockSolver<g2o::BlockSolverTraits<3, 1>>;  // block solver, parameters dims=3, residual dim=1
    // using LinearSolver = g2o::LinearSolverDense<BlockSolver::PoseMatrixType>;  // linear solver, dense
    using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;  // line solver, csparse

    // setup the solver
    // graph solver, could be Gaussian-Newton, LM, and DogLeg
    OptimizationAlgorithmLevenberg* solver =
        new OptimizationAlgorithmLevenberg(std::make_unique<BlockSolver>(std::make_unique<LinearSolver>()));
    SparseOptimizer optimizer;  // graph model
    optimizer.setAlgorithm(solver);

    // add parameter vertex to graph
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setId(0);
    v->setEstimate(Eigen::Vector3d(1, 1, 1));  // initial value
    optimizer.addVertex(v);

    // add edge(points measured) to graph
    for (size_t i = 0; i < N; ++i) {
        CurveFittingEdge* edge = new CurveFittingEdge();
        edge->setId(static_cast<int>(i));
        edge->setVertex(0, v);
        edge->setMeasurement(points[i]);
        // information matrix: the inverse of covariance matrix
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0 / (wSigma * wSigma));
        optimizer.addEdge(edge);
    }

    // begin to optimize
    auto t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);  // debug output
    optimizer.optimize(100);
    auto t2 = chrono::steady_clock::now();
    auto timeUsed = chrono::duration_cast<chrono::duration<double, std::milli>>(t2 - t1);  // milli: ms, micro: us

    // print the result
    cout << section("result") << endl;
    cout << "solve time used = " << timeUsed.count() << " ms" << endl;
    Eigen::Vector3d absEstimated = v->estimate();
    cout << "Target Curve: a * exp(-lambda * x) + b" << endl;
    cout << "Estimated Result: " << endl;
    cout << "a = " << v->estimate()(0) << endl;
    cout << "b = " << v->estimate()(1) << endl;
    cout << "lambda = " << v->estimate()(2) << endl;

    return 0;
}
