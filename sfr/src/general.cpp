#include <sfr/general.h>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace rv = std::ranges::views;
namespace ra = std::ranges;

std::vector<double> sfr::tukey(const int n0, const double mid) {
    const double m1 = n0 / 2.0, m2 = mid, m3 = n0 - mid;
    const int n = std::round(2 * std::max(m2, m3));
    auto hamming = std::vector<double>(n)
        | rv::transform([=](const int i) { return 0.54 - 0.46 * std::cos(CV_2PI * i / (n - 1)); });
    return (m2 >= m1) ?
        std::vector(hamming.begin(), hamming.begin() + n0) :
        std::vector(hamming.begin() + n - n0, hamming.end());
}

std::vector<double> sfr::center_shift(const std::vector<double>& x, const int center) {
    const int delta = x.size() / 2 - center;
    std::vector<double> out(x.size(), 0);
    if (delta > 0)
        std::copy(x.begin() + delta, x.end(), out.begin());
    else if (delta < 0)
        std::copy(x.begin(), x.end() + delta, out.begin() - delta);
    return out;
}

std::vector<double> sfr::lsf(const std::vector<double>& esf) {
    std::vector<double> diff(esf.size());
    cv::filter2D(esf, diff, CV_64F, cv::Matx<double, 1, 3>(-0.5, 0, 0.5));
    auto max = ra::max_element(diff);
    const int center = ra::distance(diff.begin(), max);
    std::vector<double> w = sfr::hamming(esf.size());
    for (size_t i = 0; i < esf.size(); i++) diff[i] = diff[i] * w[i] / *max;
    return diff;
}

std::vector<double> sfr::mtf(const std::vector<double>& lsf) {
    std::vector<std::complex<double>> fft;
    cv::dft(lsf, fft, cv::DFT_COMPLEX_OUTPUT);
    const double DC = fft[0].real();
    std::vector<double> mtf(fft.size());
    for (size_t i = 0; i < fft.size(); i++) mtf[i] = std::abs(fft[i]) / DC;
    return mtf;
}