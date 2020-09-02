//
// Created by ashutosh on 31.08.20.
//

#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat img, imgGray;
string windowName;

void GuassianSmooth(Mat &img, Mat &result) {
    // Created 5by5 discrete filter for gaussian smooth with standard deviation of 1
    float gauss_data[25] = {1, 4, 7, 4, 1,
                            4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7,
                            4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};
    // created kernel with filter data
    Mat kernel = Mat(5, 5, CV_32F, gauss_data);

    // apply filter
    filter2D(img, result, -1, kernel / 273, Point(-1, -1), 0, BORDER_DEFAULT);
    // only 4 arguments are neccessary for filter2D , other 3 are optional
    //6th is anchor in kernel is center point which shown by cv::Point(-1, -1)
    //7th is to fill target with this value before filter 0
    //8th is what to do with border elements
}

void GradientSobelFilter(Mat &Gblurred, Mat &result_sobel) {
    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2,
                        -1, 0, +1};
    float sobel_y[9] = {-1, -2, -1,
                        0, 0, 0,
                        +1, +2, +1};
    Mat kernel_x = Mat(3, 3, CV_32F, sobel_x);
    Mat kernel_y = Mat(3, 3, CV_32F, sobel_y);

    Mat result_x, result_y;

    // Sobel x filter on gaussian blurred
    filter2D(Gblurred, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    // Sobel y filter on gaussian blurred
    filter2D(Gblurred, result_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // sobel magnitude
    result_sobel = Gblurred.clone();
    for (int row = 0; row < result_sobel.rows; row++) {
        for (int col = 0; col < result_sobel.cols; col++) {
            result_sobel.at<unsigned char>(row, col) = sqrt(pow(result_x.at<unsigned char>(row, col), 2) +
                                                            pow(result_y.at<unsigned char>(row, col), 2));
        }
    }

    // visualize results
    windowName = "sobel Filter";
    imshow(windowName, result_sobel);

    int k = cv::waitKey(0); // Wait for a keystroke in the window

    if (k == 's') {
        imwrite("../Output/sobel_filter.png", result_sobel);
        cout << "Image is saved " << std::endl;
    }


}

void cornerHarris_detector() {
    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 40; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(imgGray.size(), CV_32FC1);
    cornerHarris(imgGray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    windowName = "Harris Corner Detector Response Matrix";
    imshow(windowName, dst_norm_scaled);
    int h = cv::waitKey(0); // Wait for a keystroke in the window

    if (h == 'h') {
        imwrite("../Output/Harris_detector.png", dst_norm_scaled);
        cout << "Image is saved " << std::endl;
    }

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

    //convert to grayscale
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    // Uncommnent below code block to compare gaussiansmooth result
/*    // Call for gaussian smooth

    GuassianSmooth(imgGray, result);
    string windowName = "Gaussian Window";
    imshow(windowName, result);
*/
    // Use inbuld gaussian blur function with 5by5 kernel for comparision
    GaussianBlur(imgGray, Gblurred, {5, 5}, 1.5);
    windowName = "Gaussian Window";
    imshow(windowName, Gblurred);

    int k = cv::waitKey(0); // Wait for a keystroke in the window

    if (k == 's') {
        imwrite("../Output/Gaussian_filter.png", Gblurred);
        cout << "Gaussian Blurr Image is saved " << std::endl;
    }

    // Sobel operator
    Mat result_sobel;
    GradientSobelFilter(Gblurred, result_sobel);

    // Harris corner detector
    cornerHarris_detector();

    return 0;
}


