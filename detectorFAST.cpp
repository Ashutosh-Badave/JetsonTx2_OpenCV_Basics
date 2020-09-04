//
// Created by ashutosh on 04.09.20.
//
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "BCR_adjustment.hpp"

using namespace std;
using namespace cv;

int main() {
    // load image from file and convert to grayscale
    string image_path = "../Data/KITTI/Camera/Data/0000000000.png";
    Mat imgGray;
    Mat img = imread(image_path);
    cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    int threshold = 35;   // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;     // perform non-maxima suppression on keypoints
    FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(threshold, bNMS, type);

    vector<KeyPoint> kptsFAST;
    double t = (double) getTickCount();
    detector->detect(imgGray, kptsFAST);
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "FAST with n= " << kptsFAST.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    Mat visImage = imgGray.clone();
    drawKeypoints(imgGray, kptsFAST, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "FAST Results";
    namedWindow(windowName, 2);
    imshow(windowName, visImage);
    waitKey(0);

    return 0;
}