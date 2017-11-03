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

    // 5.绘制匹配结果
    Mat imgMatch, imgGoodMatch;
    drawMatches(img1, keyPoints1, img2, keyPoints2, matches, imgMatch);
    drawMatches(img2, keyPoints1, img2, keyPoints2, goodMatches, imgGoodMatch);
    imshow("All Match", imgMatch);
    imshow("Good Match", imgGoodMatch);
    waitKey();
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

int main() {
    Mat img1 = imread("../../data/pose_estimation/1.png", IMREAD_UNCHANGED);
    Mat img2 = imread("../../data/pose_estimation/2.png", IMREAD_UNCHANGED);

    // 查找特征点
    vector<KeyPoint> keyPoints1, keyPoints2;
    vector<DMatch> goodMatches;
    findFeatureMatch(img1, img2, keyPoints1, keyPoints2, goodMatches);
    cout << "find matches size = " << goodMatches.size() << endl;

    // 估计两张图像间运动
    Mat R, t;
    poseEstimation2D2D(keyPoints1, keyPoints2, goodMatches, R, t);

    // 验证E=t^R*scale
    Mat t_x = (Mat_<double>(3, 3) <<
            0, -t.at<double>(2, 0), t.at<double>(1, 0),
            t.at<double>(2, 0), 0, -t.at<double>(0, 0),
            -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R = " << endl << t_x * R << endl;

    // 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);    // 相机内参
    for (DMatch m : goodMatches){
        Point2d pt1 = pixel2cam(keyPoints1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keyPoints2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }


    return 0;
}