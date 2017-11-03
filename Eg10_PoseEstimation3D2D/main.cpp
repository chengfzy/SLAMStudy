#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

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
        float dd = float(d) / 1000.0;
        Point2d p1 = pixel2cam(keyPoints1[m.queryIdx].pt, K);
        pts3D.emplace_back(Point3f(p1.x * dd, p1.y * dd, dd));
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

    return 0;
}