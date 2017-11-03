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


// 对极求解相机运动
void poseEstimation2D2D(const vector<KeyPoint>& keyPoints1, const vector<KeyPoint>& keyPoints2,
                        const vector<DMatch>& matches, Mat& R, Mat& t){
    // 相机内参
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 把匹配点转换成vector<Point2f>的形式
    vector<Point2f> points1, points2;
    for (int i = 0; i < matches.size(); ++i) {
        points1.emplace_back(keyPoints1[matches[i].queryIdx].pt);
        points2.emplace_back(keyPoints2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵
    Mat fundamentalMatrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental matrix is " << endl << fundamentalMatrix << endl;

    // 计算本质矩阵
    Point2d principalPoint(325.1, 249.7);       // 光心，标定值
    int focalLength = 521;                      // 焦距，标定值
    Mat essentialMatrix = findEssentialMat(points1, points2, focalLength, principalPoint, RANSAC);
    cout << "essential matrix is " << endl << essentialMatrix << endl;

    // 计算单应矩阵
    Mat homographyMatrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
    cout << "homography matrix is "  << endl << homographyMatrix << endl;

    // 从本质矩阵中恢复出旋转和平移信息
    recoverPose(essentialMatrix, points1, points2, R, t, focalLength, principalPoint);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}

// 根据对极几何求解的相机位姿，通过三角化求出特征点的空间位置
void triangulation(const vector<KeyPoint>& keyPoints1, const vector<KeyPoint>& keyPoints2,
                   const vector<DMatch>& matches, const Mat& R, const Mat& t, vector<Point3d>& points){
    Mat T1 = (Mat_<float>(3, 4) <<1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts1, pts2;
    for (DMatch m : matches){
        // 将像素坐标转换至相机坐标
        pts1.emplace_back(pixel2cam(keyPoints1[m.queryIdx].pt, K));
        pts2.emplace_back(pixel2cam(keyPoints2[m.trainIdx].pt, K));
    }

    Mat pts4d;
    triangulatePoints(T1, T2, pts1, pts2, pts4d);

    // 转换成非齐次坐标
    for (int i = 0; i < pts4d.cols; ++i) {
        Mat x = pts4d.col(i);
        x /= x.at<float>(3, 0);     // 归一化
        Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.emplace_back(p);
    }
}

int main() {
    Mat img1 = imread("../1.png", IMREAD_UNCHANGED);
    Mat img2 = imread("../2.png", IMREAD_UNCHANGED);

    // 查找特征点
    vector<KeyPoint> keyPoints1, keyPoints2;
    vector<DMatch> matches;
    findFeatureMatch(img1, img2, keyPoints1, keyPoints2, matches);
    cout << "find matches size = " << matches.size() << endl;

    // 估计两张图像间运动
    Mat R, t;
    poseEstimation2D2D(keyPoints1, keyPoints2, matches, R, t);

    // 三角化
    vector<Point3d> points;
    triangulation(keyPoints1, keyPoints2, matches, R, t, points);

    // 验证三角化点与特征点的重投影关系
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (int i = 0; i < matches.size(); ++i) {
        Point2d pt1Cam = pixel2cam(keyPoints1[matches[i].queryIdx].pt, K);
        Point2d pt1Cam3d(points[i].x / points[i].z, points[i].y / points[i].z);
        cout << "point in the first camera frame: " << pt1Cam << endl;
        cout << "point projected from 3D " << pt1Cam3d << ", d = " << points[i].z << endl;

        // 第二个图
        Point2f pt2Cam = pixel2cam(keyPoints2[matches[i].trainIdx].pt, K);
        Mat pt2Trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2Trans /= pt2Trans.at<double>(2, 0);
        cout << "point int the second camera frame: " << pt2Cam << endl;
        cout << "point reprojected from second frame: " << pt2Trans.t() << endl;
        cout << endl;
    }

    return 0;
}