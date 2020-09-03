//
// Created by ashutosh on 03.09.20.
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

Mat img, imgGray;
string windowName;

void NMS_local(Mat &dst_norm, Mat &dst_norm_scaled, int &minResponse, int &apertureSize) {
    vector<KeyPoint> keypoints;
    double maxOverlap = 0.0;

    for (int row = 0; row < dst_norm.rows; row++) {
        for (int col = 0; col < dst_norm.cols; col++) {

            int Response = static_cast<int>(dst_norm.at<float>(row, col));

            if (Response > minResponse) {
                KeyPoint Newkeypoint;
                Newkeypoint.pt = Point2f(col, row);
                Newkeypoint.size = 2 * apertureSize;
                Newkeypoint.response = Response;

                bool overlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double KPOverlap = KeyPoint::overlap(Newkeypoint, *it);
                    if (KPOverlap > maxOverlap) {
                        overlap = true;
                        if (Newkeypoint.response > (*it).response) {
                            *it = Newkeypoint;
                            break;
                        }
                    }

                }
                if (!overlap) {
                    keypoints.push_back(Newkeypoint);
                }
            }

        }
    }

    // visualize keypoints
    windowName = "NHS Results";
    Mat visImage = dst_norm_scaled.clone();
    drawKeypoints(dst_norm_scaled, keypoints, visImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow(windowName, visImage);
    int k = cv::waitKey(0); // Wait for a keystroke in the window

    if (k == 's') {
        imwrite("../Output/NHS_after_BCR_adujustment.png", visImage);
        cout << "Image is saved after NHS" << std::endl;
    }

}

void cornerHarris_detector(Mat &src) {
    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    //normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX);
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    windowName = "Harris Corner Detector Response Matrix";
    imshow(windowName, dst_norm_scaled);
    int h = cv::waitKey(0); // Wait for a keystroke in the window

    if (h == 'h') {
        imwrite("../Output/HarrisD_after_BCR_adujustment.png", dst_norm_scaled);
        cout << "Image is saved " << std::endl;
    }

    NMS_local(dst_norm, dst_norm_scaled, minResponse, apertureSize);
}

int main() {
    // Loading image from KITTI dataset
    string image_path = "../Data/KITTI/Camera/Data/0000000000.png";
    Mat result, Gblurred;
    img = imread(image_path);

    if (img.empty()) {
        cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    double contrast = 2.85; // between 1.0 - 3.0
    double brightness = 80; // between 0 - 100
    double gamma_c = 8;


    Mat img_CNB = Brightness_and_Contrast(img, brightness, contrast, gamma_c);

    //convert to grayscale
    cvtColor(img_CNB, imgGray, COLOR_BGR2GRAY);

    // Harris corner detector
    cornerHarris_detector(dst);

    return 0;
}