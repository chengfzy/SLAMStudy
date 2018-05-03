#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main() {
    Mat img1 = imread("../../data/pose_estimation/1.png", IMREAD_UNCHANGED);
    Mat img2 = imread("../../data/pose_estimation/2.png", IMREAD_UNCHANGED);

    // 初始化
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 21, 0, 2, ORB::HARRIS_SCORE, 21, 20);

    // 1.检测Oritented FAST角点位置
    orb->detect(img1, keypoints1);
    orb->detect(img2, keypoints2);

    // 2.根据角点位置计算BRIEF描述子
    orb->compute(img1, keypoints1, descriptors1);
    orb->compute(img2, keypoints2, descriptors2);

    Mat outImg;
    drawKeypoints(img1, keypoints1, outImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // 3.对两幅图像中的描述子进行匹配，使用BRIEF Hamming距离
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors1, descriptors2, matches);

    // 4.匹配点对筛选
    double minDist = 10000, maxDist = 0;
    // 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors1.rows; ++i) {
        double dist = static_cast<double>(matches[static_cast<size_t>(i)].distance);
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    printf("Max dist: %f \n", maxDist);
    printf("Min dist: %f \n", minDist);

    // 当描述子之间的距离大于两倍最小距离时，即认为匹配有误，但有时最小距离会非常小，设置一个经验值作为下限
    vector<DMatch> goodMatches;
    for (int i = 0; i < descriptors1.rows; ++i) {
        if (static_cast<double>(matches[static_cast<size_t>(i)].distance) <= max(2 * minDist, 30.0))
            goodMatches.emplace_back(matches[static_cast<size_t>(i)]);
    }

    // 5.绘制匹配结果
    Mat imgMatch, imgGoodMatch;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatch);
    drawMatches(img2, keypoints1, img2, keypoints2, goodMatches, imgGoodMatch);
    imshow("All Match", imgMatch);
    imshow("Good Match", imgGoodMatch);
    waitKey();

    return 0;
}
