#include <sfr/general.h>

std::vector<double> sfr::tukey(const int n0, const double mid) {
    const double m1 = n0 / 2.0, m2 = mid, m3 = n0 - mid;
    const int n = std::round(2 * std::max(m2, m3));
    auto hamming = std::vector<double>(n);
    for (int i = 0; i < n; i++)
        hamming[i] = 0.54 - 0.46 * std::cos(CV_2PI * i / (n - 1));
    return (m2 >= m1) ?
        std::vector(hamming.begin(), hamming.begin() + n0) :
        std::vector(hamming.begin() + n - n0, hamming.end());
}

std::vector<double> sfr::center_shift(const std::vector<double>& x, const int center) {
    const int delta = center - x.size() / 2;
    std::vector<double> out(x.size(), 0);
    if (delta > 0)
        std::copy(x.begin() + delta, x.end(), out.begin());
    else if (delta < 0)
        std::copy(x.begin(), x.end() + delta, out.begin() - delta);
    return out;
}

std::vector<double> sfr::lsf(const std::vector<double>& esf) {
    std::vector<double> diff(esf.size());
    cv::filter2D(esf, diff, CV_64F, sfr::kernel);
    double max; cv::Point maxLoc;
    cv::minMaxLoc(diff, nullptr, &max, nullptr, &maxLoc);
    diff = sfr::center_shift(diff, maxLoc.x);
    cv::multiply(diff, sfr::hamming(esf.size()), diff, 1/max);
    return diff;
}

std::vector<double> sfr::mtf(const std::vector<double>& lsf) {
    std::vector<std::complex<double>> fft;
    cv::dft(lsf, fft, cv::DFT_COMPLEX_OUTPUT);
    const double DC = fft[0].real();
    const int n = lsf.size() / sfr::factor;
    std::vector<double> mtf(n);
    for (size_t i = 0; i < n; i++) mtf[i] = std::abs(fft[i]) / DC;
    return mtf;
}

double sfr::mtf10(const std::vector<double> &mtf) {
    auto point = std::adjacent_find(mtf.begin(), mtf.end(), [](auto&& l, auto&& r) { return l > 0.1 && r < 0.1; });
    const int n = mtf.size(), i = std::distance(mtf.begin(), point);
    double k = n * (*(point+1) - *point);
    return (0.1 - *point) / k + 1.0 * i / n;
}
