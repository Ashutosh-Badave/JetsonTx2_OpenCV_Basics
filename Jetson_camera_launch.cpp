/*
Basic Code is used from JetsonHacks and did few modifications for better use.
Author : Ashutosh Badave
*/

#include <iostream>
#include <opencv2/opencv.hpp>




std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main() {
    // Options
    int WIDTH = 1920;
    int HEIGHT = 1080;
    int FPS = 30;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Connection failed";
        return -1;
    }

    // View video
    cv::Mat frame;
    while (1) {
        cap >> frame;  // Get a new frame from camera

        // Display frame
        imshow("Display window", frame);
        cv::waitKey(1); //needed to show frame
    }
}
