#include <iostream>
#include <sfr/general.h>
#include <sfr/slanted.h>
#include <sfr/cylinder.h>
#include <opencv2/core/utils/logger.hpp>
#include <numeric>


void test_slanted(){
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
    std::cout << "=============================" << '\n';
    std::cout << sfr::mtf10(mtf) << '\n';
}

void test_cylinder(){
    std::string path = R"(D:\Code\python\mtf_final\data\000-4.tif)"; // 直径4mm
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::cout << img.size() << '\n';
    auto circle = sfr::circle_fit(img, 100);
    auto [cx, cy, r] = circle;
    std::cout << "cx:" << cx << " cy:" << cy <<  " r:" << r << '\n';
    const float roi = 8, binsize = 1. / 32;
    auto esf = sfr::esf(img, {cx, cy, r}, roi, binsize);
    std::cout << std::max_element(esf.begin(), esf.end(), [](auto lop, auto rop) { return lop.y < rop.y;})->x << '\n';
    // for (auto& val : esf) std::cout << val.x << ',' << val.y << '\n';
    std::cout << "===========lsf================" << '\n';
    auto lsf = sfr::lsf(esf);
    for (auto& val : lsf) std::cout << val.x << ',' << val.y << '\n';
    std::cout << "===========mtf================" << '\n';
    auto mtf = sfr::mtf(lsf, binsize * 0.5 * 4 / r * 25);
    for (auto& val : mtf) std::cout << val.x << ',' << val.y << '\n';
    std::cout << "mtf10:" << sfr::mtf10(mtf) << '\n';
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    test_slanted();
    // test_cylinder();

}