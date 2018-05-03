#include <chrono>
#include <iostream>
#include <memory>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace g2o;

// camera intrinsics K = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
const Mat kCameraMatrix = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

// find keyPoints and matches
void findFeatureMatch(const Mat& img1, const Mat& img2, vector<KeyPoint>& keyPoints1, vector<KeyPoint>& keyPoints2,
                      vector<DMatch>& goodMatches, bool showMatchImage = true) {
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    // 1.检测Oritented FAST角点位置
    detector->detect(img1, keyPoints1);
    detector->detect(img2, keyPoints2);

    // 2.根据角点位置计算BRIEF描述子
    descriptor->compute(img1, keyPoints1, descriptors1);
    descriptor->compute(img2, keyPoints2, descriptors2);

    // 3.对两幅图像中的描述子进行匹配，使用BRIEF Hamming距离
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // 4.匹配点对筛选
    double minDist = 10000, maxDist = 0;
    // 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors1.rows; ++i) {
        double dist = static_cast<double>(matches[static_cast<size_t>(i)].distance);
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }
    cout << "Max dist  = " << maxDist << endl;
    cout << "Min dist  = " << minDist << endl;
    // 当描述子之间的距离大于两倍最小距离时，即认为匹配有误，但有时最小距离会非常小，设置一个经验值作为下限
    for (size_t i = 0; i < static_cast<size_t>(descriptors1.rows); ++i) {
        if (static_cast<double>(matches[i].distance) <= max(2 * minDist, 30.0)) {
            goodMatches.emplace_back(matches[i]);
        }
    }

    // print match result
    cout << "all matches size = " << matches.size() << endl;
    cout << "goold matches size = " << goodMatches.size() << endl;

    // 5.Show match image
    if (showMatchImage) {
        Mat imgMatch, imgGoodMatch;
        drawMatches(img1, keyPoints1, img2, keyPoints2, matches, imgMatch);
        drawMatches(img2, keyPoints1, img2, keyPoints2, goodMatches, imgGoodMatch);
        imshow("All Match", imgMatch);
        imshow("Good Match", imgGoodMatch);
        waitKey();
    }
}

// convert points(pixel) from image frame to point (P_c) in normalized camera frame, P_uv = K * P_c
Point2d pixel2cam(const Point2d& p, const Mat& K) {
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

// estimate pose based on points 3D-3D
void poseEstimation3D3D(const vector<Point3f>& pts1, const vector<Point3f>& pts2, Mat& R, Mat& t) {
    // center of mass p1, p2
    Point3f p1, p2;
    const size_t kN = pts1.size();
    for (size_t i = 0; i < kN; ++i) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 /= static_cast<float>(kN);
    p2 /= static_cast<float>(kN);

    // remove center position, q1 = pts1 - p1, q2 = pts2 - p2
    vector<Point3f> q1(kN), q2(kN);
    for (size_t i = 0; i < kN; ++i) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute W = Sum(q1 * q2^T)
    Matrix3d W = Matrix3d::Zero();
    for (size_t i = 0; i < kN; ++i) {
        W += Vector3d(static_cast<double>(q1[i].x), static_cast<double>(q1[i].y), static_cast<double>(q1[i].z)) *
             Vector3d(static_cast<double>(q2[i].x), static_cast<double>(q2[i].y), static_cast<double>(q2[i].z))
                 .transpose();
    }
    cout << "W = " << W << endl;

    // SVD on W
    JacobiSVD<Matrix3d> svd(W, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();
    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    // compute R = U * V^T, t = p1 - R * p2;
    Matrix3d R_ = U * V.transpose();
    Vector3d t_ = Vector3d(p1.x, p1.y, p1.z) - R_ * Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2), R_(1, 0), R_(1, 1), R_(1, 2), R_(2, 0), R_(2, 1),
         R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

// g2o 3D-3D edge
class EdgeProjectXYZRGBDPose : public g2o::BaseUnaryEdge<3, Vector3d, g2o::VertexSE3Expmap> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectXYZRGBDPose(const Vector3d& point) : _point(point) {}

    virtual void computeError() {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map(_point);
    }

    virtual void linearizeOplus() {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Vector3d xyzTrans = T.map(_point);
        double x = xyzTrans[0], y = xyzTrans[1], z = xyzTrans[2];

        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = -z;
        _jacobianOplusXi(0, 2) = y;
        _jacobianOplusXi(0, 3) = -1;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = 0;

        _jacobianOplusXi(1, 0) = z;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(1, 2) = -x;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -1;
        _jacobianOplusXi(1, 5) = 0;

        _jacobianOplusXi(2, 0) = -y;
        _jacobianOplusXi(2, 1) = x;
        _jacobianOplusXi(2, 2) = 0;
        _jacobianOplusXi(2, 3) = 0;
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = -1;
    }

    bool read(istream&) {}
    bool write(ostream&) const {}

   protected:
    Vector3d _point;
};

// Bundle Adjustment
void bundleAdjust(const vector<Point3f>& pts1, const vector<Point3f>& pts2, Mat& R, Mat& t) {
#if 0
    // initialize
    unique_ptr<BlockSolver_6_3::LinearSolverType> linearSolver{
        new LinearSolverCSparse<BlockSolver_6_3::PoseMatrixType>()};
    OptimizationAlgorithmLevenberg* solver =
        new OptimizationAlgorithmLevenberg(make_unique<BlockSolver_6_3>(move(linearSolver)));
    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex: camera pose
    VertexSE3Expmap* pose = new VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(SE3Quat(Matrix3d::Identity(), Vector3d(0, 0, 0)));
    optimizer.addVertex(pose);

    // edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPose*> edges;
    for (size_t i = 0; i < pts1.size(); ++i) {
        EdgeProjectXYZRGBDPose* edge = new EdgeProjectXYZRGBDPose(Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement(Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Matrix3d::Identity() * 1e4);
        optimizer.addEdge(edge);
        edges.push_back(edge);
        ++index;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeUsed = chrono::duration_cast<chrono::seconds>(t2 - t1);
    cout << "optimization costs time: " << timeUsed.count() << " seconds." << endl;

    cout << endl << "after optimization: " << endl;
    cout << "T = " << endl << Isometry3d(pose->estimate()).matrix() << endl;
#endif
}

int main() {
    // load BGR and depth image
    Mat img1 = imread("../../data/pose_estimation/1.png", IMREAD_UNCHANGED);
    Mat img2 = imread("../../data/pose_estimation/2.png", IMREAD_UNCHANGED);
    Mat depth1 = imread("../../data/pose_estimation/1_depth.png", IMREAD_UNCHANGED);
    Mat depth2 = imread("../../data/pose_estimation/2_depth.png", IMREAD_UNCHANGED);

    // find features and match
    vector<KeyPoint> keyPoints1, keyPoints2;
    vector<DMatch> matches;
    findFeatureMatch(img1, img2, keyPoints1, keyPoints2, matches, false);
    cout << "find matches size = " << matches.size() << endl;

    // construct 3D-3D points pair
    vector<Point3f> pts1;
    vector<Point3f> pts2;
    for (auto m : matches) {
        auto pt1 = keyPoints1[static_cast<size_t>(m.queryIdx)].pt;  // keypoint points
        auto pt2 = keyPoints2[static_cast<size_t>(m.trainIdx)].pt;
        ushort d1 = depth1.ptr<unsigned short>(static_cast<int>(pt1.y))[static_cast<int>(pt1.x)];
        ushort d2 = depth2.ptr<unsigned short>(static_cast<int>(pt2.y))[static_cast<int>(pt2.x)];
        if (0 == d1 || 0 == d2) continue;  // bypass bad depth

        float dd1 = d1 / 1000.0f;
        float dd2 = d2 / 1000.0f;
        Point2d p1 = pixel2cam(pt1, kCameraMatrix);
        Point2d p2 = pixel2cam(pt2, kCameraMatrix);
        pts1.emplace_back(Point3f(static_cast<float>(p1.x) * dd1, static_cast<float>(p1.y) * dd1, dd1));
        pts2.emplace_back(Point3f(static_cast<float>(p2.x) * dd2, static_cast<float>(p2.y) * dd2, dd2));
    }
    cout << "3D-3D pairs = " << pts1.size() << endl;

    // estiamte camera motion using 3D-3D points pair and ICP method
    cout << "================================ Pose Estimation 3D-3D using ICP ================================" << endl;
    Mat R, t;
    poseEstimation3D3D(pts1, pts2, R, t);
    cout << "ICP via SVD" << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    // bundle adjust to optimize camera motion parameters
    cout << endl << "bundle adjustment" << endl;
    bundleAdjust(pts1, pts2, R, t);

    // verify p1 = R * p2 + t
    for (size_t i = 0; i < 5; ++i) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "R * p2 + t = " << (R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t).t() << endl;
    }

    return 0;
}
