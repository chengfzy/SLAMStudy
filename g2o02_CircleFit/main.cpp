/*
 * Using g2o for circle fitting
 * Parameters: center, radius
 */

#include <iostream>
#include <vector>
#include "Common.hpp"
#include "Eigen/Cholesky"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/stuff/sampler.h"

using namespace std;
using namespace g2o;
using namespace Eigen;

// compute the error of the input paramter for the points
double errorOfSolution(const vector<Vector2d>& points, const Vector3d& circleParams) {
    const Vector2d center = circleParams.head<2>();
    const double radius = circleParams(2);
    double error{0};

    for (auto& p : points) {
        double d = (p - center).norm() - radius;
        error += d * d;
    }
    return error;
}

/**
 * @brief Circle Vertex: the center x, y and radius r
 */
class CircleVertex : public BaseVertex<3, Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(istream&) override {
        cout << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    bool write(ostream&) const override {
        cout << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    void setToOriginImpl() override { _estimate << 0, 0, 0; }

    void oplusImpl(const double* v) override { _estimate += Eigen::Map<const Eigen::Vector3d>(v); }
};

/**
 * @brief Measurement for a point on the circle
 * The error function computes the distance of the point to the center - the radius of the circle, the measurement is
 * the point which is on the circle
 */
class CircleEdge : public BaseUnaryEdge<1, Vector2d, CircleVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(istream&) override {
        cout << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    bool write(ostream&) const override {
        cout << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    void computeError() override {
        const CircleVertex* v = static_cast<const CircleVertex*>(vertex(0));
        const Vector2d& center = v->estimate().head<2>();
        const double& radius = v->estimate()(2);

        _error(0) = (measurement() - center).norm() - radius;
    }
};

int main(int argc, char* argv[]) {
    Vector2d center(4.0, 2.0);
    double radius{2.0};
    const size_t N{100};  // data length

    // generate random data
    cout << section("generate random data") << endl;
    vector<Vector2d> points(N);
    for (size_t i = 0; i < N; ++i) {
        double r = g2o::Sampler::gaussRand(radius, 0.05);
        double angle = g2o::Sampler::uniformRand(0.0, 2.0 * M_PI);
        points[i].x() = center.x() + r * cos(angle);
        points[i].y() = center.y() + r * sin(angle);
    }

    // type definitions
    using BlockSolver = g2o::BlockSolver<BlockSolverTraits<3, 1>>;
    using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;

    // setup the solver
    SparseOptimizer optimizer;
    OptimizationAlgorithmLevenberg* optAlg =
        new OptimizationAlgorithmLevenberg(std::make_unique<BlockSolver>(std::make_unique<LinearSolver>()));
    optimizer.setAlgorithm(optAlg);

    // add parameter vertex to graph
    CircleVertex* v = new CircleVertex();
    v->setId(0);
    v->setEstimate(Vector3d(3.0, 3.0, 3.0));
    optimizer.addVertex(v);

    // add points(edge) to graph
    for (size_t i = 0; i < points.size(); ++i) {
        CircleEdge* edge = new CircleEdge();
        edge->setId(static_cast<int>(i));
        edge->setVertex(0, v);
        edge->setMeasurement(points[i]);
        edge->setInformation(Matrix<double, 1, 1>::Identity());
        optimizer.addEdge(edge);
    }

    // perform the optimization
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(100);

    // print the result
    cout << section("Optimization Result") << endl;
    cout << "center = (" << v->estimate()(0) << ", " << v->estimate()(1) << "), radius = " << v->estimate()(2) << endl;
    cout << "error = " << errorOfSolution(points, v->estimate()) << endl;

    // solved by linear least squares
    // Let (a,b) be the center of the circle and r the radius of the circle, for a point (x,y) on the circles, we have
    // (x - a)^2 + (y - b)^2 = r^2
    // This leads to
    // [-2x, -2y, 1]^T * [a, b, a^2 + b^2 - r^2] = -x^2 - y^2
    // This could be solve by Ax = b, and a, b, r then could be recovered from x
    Matrix<double, N, 3> A;
    Matrix<double, N, 1> b;
    for (size_t i = 0; i < points.size(); ++i) {
        A(i, 0) = -2 * points[i].x();
        A(i, 1) = -2 * points[i].y();
        A(i, 2) = 1;
        b(i) = -pow(points[i].x(), 2) - pow(points[i].y(), 2);
    }
    Vector3d x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // calculate r
    x(2) = sqrt(pow(x(0), 2) + pow(x(1), 2) - x(2));
    cout << section("Linear Least Square Solution") << endl;
    cout << "center = (" << x(0) << ", " << x(1) << "), radius = " << x(2) << endl;
    cout << "error = " << errorOfSolution(points, x) << endl;
}