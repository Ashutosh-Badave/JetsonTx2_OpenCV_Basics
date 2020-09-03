//
// Created by ashutosh on 03.09.20.
//

#ifndef OPENCV_BASICS_BCR_ADJUSTMENT_HPP
#define OPENCV_BASICS_BCR_ADJUSTMENT_HPP

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

Mat gammaCorrection(const Mat &source, const double gamma_) {
    CV_Assert(gamma_ >= 0);
    //! [changing-contrast-brightness-gamma-correction]
    Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

    Mat res = source.clone();
    LUT(source, lookUpTable, res);

    //Mat img_gamma_corrected;
    //hconcat(source, res, img_gamma_corrected);
    //imshow("Gamma correction", img_gamma_corrected);
    return res;
}

Mat
Brightness_and_Contrast(Mat &img, const double &brightness = 0, const double &contrast = 0, const double &gamma_c = 0) {

    Mat new_image = Mat::zeros(img.size(), img.type());

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            for (int chnl = 0; chnl < img.channels(); chnl++) {
                new_image.at<Vec3b>(row, col)[chnl] = saturate_cast<unsigned char>(
                        img.at<Vec3b>(row, col)[chnl] * contrast + brightness);
            }
        }
    }

    Mat img_gammaC = gammaCorrection(new_image, gamma_c);

    return img_gammaC;
}

#endif //OPENCV_BASICS_BCR_ADJUSTMENT_HPP
