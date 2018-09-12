/*
 * Triangulation to obtain position of map point
 */
#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

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
        drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, imgGoodMatch);
        imshow("All Match", imgMatch);
        imshow("Good Match", imgGoodMatch);
        waitKey();
    }
}

// convert points from image frame(pixel) to normalized camera frame(pc), puv = K * pc
Point2d pixel2cam(const Point2d& p, const Mat& K) {
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

// estimate pose 2D-2D, using epipolar geometry constraints
void poseEstimation2D2D(const vector<KeyPoint>& keyPoints1, const vector<KeyPoint>& keyPoints2,
                        const vector<DMatch>& matches, Mat& R, Mat& t) {
    // 把匹配点转换成vector<Point2f>的形式
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); ++i) {
        points1.emplace_back(keyPoints1[static_cast<size_t>(matches[i].queryIdx)].pt);
        points2.emplace_back(keyPoints2[static_cast<size_t>(matches[i].trainIdx)].pt);
    }

    // 计算基础矩阵
    Mat fundamentalMatrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "Fundamental Matrix F = " << endl << fundamentalMatrix << endl;

    // 计算本质矩阵
    Mat essentialMatrix = findEssentialMat(points1, points2, kCameraMatrix, RANSAC);
    cout << "Essential Matrix E = " << endl << essentialMatrix << endl;

    // 计算单应矩阵
    Mat homographyMatrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
    cout << "Homography Natrix H = " << endl << homographyMatrix << endl;

    // 从本质矩阵中恢复出旋转和平移信息
    recoverPose(essentialMatrix, points1, points2, kCameraMatrix, R, t);
    cout << "Rotation R = " << endl << R << endl;
    cout << "Translation t = " << endl << t << endl;
}

// triangulation to obtain 3D position in world from keypoints R and t
void triangulation(const vector<KeyPoint>& keyPoints1, const vector<KeyPoint>& keyPoints2,
                   const vector<DMatch>& matches, Mat& R, Mat& t, vector<Point3d>& points) {
    // projection matrix
    Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0), R.at<double>(2, 0),
              R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    // convert points(pixel) from image frame to normalized camera frame
    vector<Point2f> pts1, pts2;
    for (auto m : matches) {
        pts1.emplace_back(pixel2cam(keyPoints1[static_cast<size_t>(m.queryIdx)].pt, kCameraMatrix));
        pts2.emplace_back(pixel2cam(keyPoints2[static_cast<size_t>(m.trainIdx)].pt, kCameraMatrix));
    }

    // triangulation
    Mat pts4d;
    triangulatePoints(T1, T2, pts1, pts2, pts4d);

    // convert to inhomogeneous coordinates, please note pts4D is CV_32F type
    for (int i = 0; i < pts4d.cols; ++i) {
        Mat x = pts4d.col(i);
        x /= static_cast<double>(x.at<float>(3, 0));
        points.emplace_back(Point3d(static_cast<double>(x.at<float>(0, 0)), static_cast<double>(x.at<float>(1, 0)),
                                    static_cast<double>(x.at<float>(2, 0))));
    }
}

// test pose estimation using 2D-2D features
void testPoseEstimate2D2D(const Mat& img1, const Mat& img2) {
    cout << "=================================== Pose Estimation 2D-2D ===================================" << endl;
    // find features and match
    vector<KeyPoint> keyPoints1, keyPoints2;
    vector<DMatch> matches;
    findFeatureMatch(img1, img2, keyPoints1, keyPoints2, matches, false);
    cout << "find matches size = " << matches.size() << endl;

    // estimate R, t from two images
    Mat R, t;
    poseEstimation2D2D(keyPoints1, keyPoints2, matches, R, t);

    // check epipolar geometry constrains: x2^T * t^ * R * x1 = 0
    Mat t_x = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0), t.at<double>(2, 0), 0,
               -t.at<double>(0, 0), -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout << "t^R = " << endl << t_x * R << endl;
    cout << "------ check epipolar geometry constrains: x2^T * t^ * R * x1 = 0 ------" << endl;
    for (DMatch m : matches) {
        Point2d pt1 = pixel2cam(keyPoints1[static_cast<size_t>(m.queryIdx)].pt, kCameraMatrix);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keyPoints2[static_cast<size_t>(m.trainIdx)].pt, kCameraMatrix);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
}

// test triangulation to obtain 3D points from two images
void testTriangulation(const Mat& img1, const Mat& img2) {
    cout << "=================================== Triangulation ===================================" << endl;
    // find features and match
    vector<KeyPoint> keyPoints1, keyPoints2;
    vector<DMatch> matches;
    findFeatureMatch(img1, img2, keyPoints1, keyPoints2, matches, true);
    cout << "find matches size = " << matches.size() << endl;

    // estimate R, t from two images
    Mat R, t;
    poseEstimation2D2D(keyPoints1, keyPoints2, matches, R, t);

    // triangulation
    vector<Point3d> points;
    triangulation(keyPoints1, keyPoints2, matches, R, t, points);

    // check the reprojection error between triangulated points
    for (size_t i = 0; i < matches.size(); ++i) {
        Point2d pt1Cam = pixel2cam(keyPoints1[static_cast<size_t>(matches[i].queryIdx)].pt, kCameraMatrix);
        Point2d pt1Cam3d(points[i].x / points[i].z, points[i].y / points[i].z);
        cout << "pc1 = " << pt1Cam << ", pw = " << points[i] << endl;
    }
}

int main(int argc, char* argv[]) {
    // read image
    Mat img1 = imread("../../data/pose_estimation/1.png", IMREAD_UNCHANGED);
    Mat img2 = imread("../../data/pose_estimation/2.png", IMREAD_UNCHANGED);

    // triangulation
    testTriangulation(img1, img2);

    return 0;
}
