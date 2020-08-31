//
// Created by ashutosh on 31.08.20.
//

#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void GuassianSmooth() {


}

int main() {
    // Loading image from KITTI dataset
    string image_path = "../Data/KITTI/Camera/Data/0000000000.png";
    cv::Mat img;

    img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    string windowName = "Output";
    cv::imshow(windowName, img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if (k == 's') {
        cv::imwrite("KITTI.png", img);
        std::cout << "Image is saved " << std::endl;
    }
    return 0;
}

