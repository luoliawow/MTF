#include <iostream>
#include <sfr/general.h>
#include <sfr/slanted.h>

int main() {
    std::string path =  R"(D:\Code\cpp\Implements\sfrmat5_dist\Example_Images\Test_edge1_mono.tif)";
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    auto [k, b] = sfr::line_fit(img);
    const double delfac = std::sqrt(1 / (1 + k * k));
    const int nrow = std::round(std::floor(std::abs(img.rows * k)) / std::abs(k));
    cv::Mat roi = img(cv::Range(0, nrow), cv::Range(0, img.cols));
    auto esf = sfr::esf(roi, k);
    auto lsf = sfr::lsf(esf);
    auto mtf = sfr::mtf(lsf);
    for (auto& val : mtf) std::cout << val << '\n';
}