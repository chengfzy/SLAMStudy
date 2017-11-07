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


// Bundle Adjustment
void bundleAdjust(const vector<Point3f>& points3D, const vector<Point2f>& points2D, const Mat& K, Mat& R, Mat& t){
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;   // pose维度为6,landmark维度为3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();      // 线性方程求解器
    Block* solverPtr = new Block(linearSolver);                    // 矩形块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solverPtr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();        // camera pose
    Eigen::Matrix3d RMat;
    RMat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(RMat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    optimizer.addVertex(pose);

    // landmarks
    int index = 1;
    for (const Point3f p : points3D){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);   // g2o必须设置marg
        optimizer.addVertex(point);
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // edges
    index = 1;
    for (const Point2f p : points2D){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        ++index;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(200);
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
    vector<Point3f> pts3D;
    vector<Point2f> pts2D;

    for (DMatch m : matches){
        ushort d = depth1.ptr<unsigned short>(int(keyPoints1[m.queryIdx].pt.y))[int(keyPoints1[m.queryIdx].pt.x)];
        if (0 == d) continue;       // bad depth
        float dd = static_cast<float>(d) / 1000.0f;
        Point2d p1 = pixel2cam(keyPoints1[m.queryIdx].pt, K);
        pts3D.emplace_back(Point3f(static_cast<float>(p1.x * dd), static_cast<float>(p1.y * dd), dd));
        pts2D.emplace_back(keyPoints2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts3D.size() << endl;
    Mat r, t;
    // 调用OpenCV的Pnp求解，可选择EPnP，DLS等方法
    solvePnP(pts3D, pts2D, K, Mat(), r, t, false);
    Mat R;
    Rodrigues(r, R);        // r为旋转微量形式，用Rodrigues公式转换为矩阵

    cout << "R = " << endl << R << endl;
    cout << "t = " << endl << t << endl;

    cout << "calling bundle adjustment" << endl;
    bundleAdjust(pts3D, pts2D, K, R, t);

    return 0;
}