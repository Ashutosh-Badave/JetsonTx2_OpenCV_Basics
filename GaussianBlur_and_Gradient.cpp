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

void GuassianSmooth(Mat &img, Mat &result) {
    // Created 5by5 discrete filter for gaussian smooth with standard deviation of 1
    float gauss_data[25] = {1, 4, 7, 4, 1,
                            4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7,
                            4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};
    // created kernel with filter data
    Mat kernel = cv::Mat(5, 5, CV_32F, gauss_data);

    // apply filter
    filter2D(img, result, -1, kernel / 273, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    // only 4 arguments are neccessary for filter2D , other 3 are optional
    //6th is anchor in kernel is center point which shown by cv::Point(-1, -1)
    //7th is to fill target with this value before filter 0
    //8th is what to do with border elements
}

int main() {
    // Loading image from KITTI dataset
    string image_path = "../Data/KITTI/Camera/Data/0000000000.png";
    cv::Mat img, result, result1;

    img = imread(image_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    // Call for gaussian smooth
    GuassianSmooth(img, result);
    string windowName = "Gaussian Window1";
    imshow(windowName, result);

    // Use inbuld gaussian blur function with 5by5 kernel for comparision
    GaussianBlur(img, result1, {5, 5}, 0);
    windowName = "Gaussian Window2";
    imshow(windowName, result1);
    int k = cv::waitKey(0); // Wait for a keystroke in the window

    if (k == 's') {
        imwrite("KITTI.png", result1);
        cout << "Image is saved " << std::endl;
    }
    return 0;
}

