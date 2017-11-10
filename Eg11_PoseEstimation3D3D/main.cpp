#include <iostream>
#include <memory>
#include <chrono>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

using namespace std;
using namespace cv;


/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

// find keyPoints and matches
void findFeatureMatch(const Mat& img1, const Mat& img2,
                      vector<KeyPoint>& keyPoints1, vector<KeyPoint>& keyPoints2, vector<DMatch>& goodMatches){
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    // 1.检测Oritented FAST角点位置
    detector->detect(img1, keyPoints1);
    detector->detect(img2, keyPoints2);

    // 2.根据角点位置计算BRIEF描述子
    descriptor->compute(img1, keyPoints1, descriptors1);
    descriptor->compute(img2, keyPoints2, descriptors2);

    Mat outImg;
    drawKeypoints(img1, keyPoints1, outImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // 3.对两幅图像中的描述子进行匹配，使用BRIEF Hamming距离
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // 4.匹配点对筛选
    double minDist = 10000, maxDist = 0;
    // 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors1.rows; ++i) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    printf("Max dist: %f \n", maxDist);
    printf("Min dist: %f \n", minDist);

    // 当描述子之间的距离大于两倍最小距离时，即认为匹配有误，但有时最小距离会非常小，设置一个经验值作为下限
    for (int i = 0; i < descriptors1.rows; ++i) {
        if (matches[i].distance <= max(2 * minDist, 30.0))
            goodMatches.emplace_back(matches[i]);
    }
}

Point2d pixel2cam(const Point2d& p, const Mat& K){
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

// estimate pose based on points 3D-3D
void poseEstimation3D3D(const vector<Point3f>& pts1, const vector<Point3f>& pts2, Mat& R, Mat& t){
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; ++i) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 /= N;
    p2 /= N;

    vector<Point3f> q1(N), q2(N);       // remove the center
    for (int i = 0; i < N; ++i) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1 * q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x ,q2[i].y, q2[i].z).transpose();
    }
    cout << "W = " << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    Eigen::Matrix3d R_ = U * V.transpose();
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2));

    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

// g2o edge
class EdgeProjectXYZRGBDPose : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectXYZRGBDPose(const Eigen::Vector3d& point) : _point(point){}

    virtual void computeError(){
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map(_point);
    }

    virtual void linearizeOplus(){
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyzTrans = T.map(_point);
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

    bool read(istream& in) {}
    bool write(ostream& out) const {}
protected:
    Eigen::Vector3d _point;
};

// Bundle Adjustment
void bundleAdjust(const vector<Point3f>& pts1, const vector<Point3f>& pts2, Mat& R, Mat& t){
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;   // pose维度为6,landmark维度为3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();      // 线性方程求解器
    Block* solverPtr = new Block(linearSolver);                    // 矩形块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solverPtr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();        // camera pose
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0)));
    optimizer.addVertex(pose);

    // edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPose*> edges;
    for (int i = 0; i < pts1.size(); ++i) {
        EdgeProjectXYZRGBDPose* edge = new EdgeProjectXYZRGBDPose(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);
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
    cout << "T = " << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}




int main() {
    Mat img1 = imread("../../data/pose_estimation/1.png", IMREAD_UNCHANGED);
    Mat img2 = imread("../../data/pose_estimation/2.png", IMREAD_UNCHANGED);

    // 查找特征点
    vector<KeyPoint> keyPoints1, keyPoints2;
    vector<DMatch> matches;
    findFeatureMatch(img1, img2, keyPoints1, keyPoints2, matches);
    cout << "find matches size = " << matches.size() << endl;

    // 建立3D点
    Mat depth1 = imread("../../data/pose_estimation/1_depth.png", IMREAD_UNCHANGED);
    Mat depth2 = imread("../../data/pose_estimation/2_depth.png", IMREAD_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1;
    vector<Point3f> pts2;

    for (DMatch m : matches){
        ushort d1 = depth1.ptr<unsigned short>(int(keyPoints1[m.queryIdx].pt.y))[int(keyPoints1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keyPoints2[m.trainIdx].pt.y))[int(keyPoints2[m.trainIdx].pt.y)];

        if (0 == d1 || 0 == d2) continue;       // bad depth

        Point2d p1 = pixel2cam(keyPoints1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keyPoints2[m.trainIdx].pt, K);
        float dd1 = static_cast<float>(d1 / 1000.0);
        float dd2 = static_cast<float>(d2 / 1000.0);

        pts1.emplace_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.emplace_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3D-3D pairs: " << pts1.size() << endl;
    Mat R, t;
    poseEstimation3D3D(pts1, pts2, R, t);
    cout << "ICP via SVD result: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    cout << "calling bundle adjustment" << endl;
    bundleAdjust(pts1, pts2, R, t);

    // verify p1 = R * p2 + t
    for (int i = 0; i < 5; ++i) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "R * p2 + t = " << (R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t).t() << endl;
    }

    return 0;
}