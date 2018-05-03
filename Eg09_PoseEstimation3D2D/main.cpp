#include <chrono>
#include <iostream>
#include <memory>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
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

// Bundle Adjustment
void bundleAdjust(const vector<Point3f>& points3D, const vector<Point2f>& points2D, const Mat& K, Mat& R, Mat& t) {
    cout << "=========================== Pose Estimation 3D-2D using Bundle Adjust ===========================" << endl;
    // 初始化g2o
    //    using Block = BlockSolver<BlockSolverTraits<6, 3>>;  // pose维度为6,landmark维度为3
    unique_ptr<BlockSolver_6_3::LinearSolverType> linearSolver{
        new LinearSolverCholmod<BlockSolver_6_3::PoseMatrixType>()};
    OptimizationAlgorithmLevenberg* solver =
        new OptimizationAlgorithmLevenberg(make_unique<BlockSolver_6_3>(move(linearSolver)));
    SparseOptimizer optimizer;  // graph model
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);  // debug output

    // add vertex(pose) to graph
    Eigen::Matrix3d RMat;
    RMat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(1, 0), R.at<double>(1, 1),
        R.at<double>(1, 2), R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
#if 0
    VertexSE3Expmap* pose = new VertexSE3Expmap();  // camera pose
    pose->setId(0);
    pose->setEstimate(SE3Quat(RMat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    optimizer.addVertex(pose);

    // add edge(3D landmarks) to graph
    int index{1};
    for (auto p : points3D) {
        VertexSBAPointXYZ* point = new VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(
            Eigen::Vector3d(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)));
        point->setMarginalized(true);  // g2o必须设置marg
        optimizer.addVertex(point);
    }

    // parameter: camera intrinsics
    CameraParameters* camera =
        new CameraParameters(K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // add edges(2D measurments) to graph
    index = 1;
    for (auto p : points2D) {
        EdgeProjectXYZ2UV* edge = new EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        ++index;
    }

    // optimize
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(200);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeUsed = chrono::duration_cast<chrono::seconds>(t2 - t1);
    cout << "optimization costs time: " << timeUsed.count() << " seconds." << endl;

    // print out result
    cout << endl << "after optimization: " << endl;
    cout << "T = " << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
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

    // construct 3D-2D points pair
    vector<Point3f> pts3D;  // 3D points in image1
    vector<Point2f> pts2D;  // 2D points in image2
    for (auto m : matches) {
        auto pt = keyPoints1[static_cast<size_t>(m.queryIdx)].pt;  // keypoint points
        ushort d = depth1.ptr<unsigned short>(static_cast<int>(pt.y))[static_cast<int>(pt.x)];
        if (0 == d) continue;  // bypass bad depth

        float dd = d / 1000.0f;
        Point2d p1 = pixel2cam(pt, kCameraMatrix);
        pts3D.emplace_back(Point3f(static_cast<float>(p1.x) * dd, static_cast<float>(p1.y) * dd, dd));
        pts2D.emplace_back(keyPoints2[static_cast<size_t>(m.trainIdx)].pt);
    }
    cout << "3D-2D pairs = " << pts3D.size() << endl;

    // estiamte camera motion using 3D-2D points pair and EPnP method
    cout << "================================ Pose Estimation 3D-2D using PnP ================================" << endl;
    // calculate camera motion R, t. could use another method, like EPnP, DLS etc.
    Mat R, r, t;
    solvePnP(pts3D, pts2D, kCameraMatrix, Mat(), r, t, false, SOLVEPNP_EPNP);
    Rodrigues(r, R);  // convert r to R, Vec to Mat
    cout << "R = " << endl << R << endl;
    cout << "t = " << t << endl;

    // bundle adjust to optimize camera motion parameters
    bundleAdjust(pts3D, pts2D, kCameraMatrix, R, t);

    return 0;
}
