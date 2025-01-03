#ifndef GENERAL_H
#define GENERAL_H
#include <vector>
#include <opencv2/opencv.hpp>

namespace sfr {
    const int factor = 4;
    const cv::Matx<double, 1, 3> kernel(-0.5, 0, 0.5);
    std::vector<double> tukey(const int n0, const double mid);
    inline std::vector<double> hamming(const int n) { return tukey(n, n/2); }
    std::vector<double> center_shift(const std::vector<double>& x, const int center);
    std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree);
    std::vector<double> polyval(const std::vector<double>& x, const std::vector<double>& coeff);
}


#endif //GENERAL_H
