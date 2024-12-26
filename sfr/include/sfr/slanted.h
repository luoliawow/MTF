#ifndef SLANTED_H
#define SLANTED_H
#include <tuple>
#include <opencv2/opencv.hpp>

namespace sfr {
    std::tuple<double, double> line_fit(const cv::Mat& mat);
    std::vector<double> esf(const cv::Mat& mat, const double k);
    std::vector<double> lsf(const std::vector<double>& esf);
    std::vector<double> mtf(const std::vector<double>& lsf);
    double mtf10(const std::vector<double>& mtf);
}

#endif //SLANTED_H
