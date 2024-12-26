#include <sfr/slanted.h>
#include <sfr/general.h>

std::tuple<double, double> centroid_fit(const cv::InputArray img) {
    cv::Mat mat = img.getMat();
    std::vector<cv::Point2d> edge;
    for (int i = 0; i < mat.rows; ++i) {
        const double* ptr = mat.ptr<double>(i);
        double sum = 0, sum_y = 0;
        for (int j = 0; j < mat.cols; ++j) {
            sum += ptr[j]; sum_y += ptr[j] * j;
        }
        edge.emplace_back(i, sum_y / sum - 0.5);
    }
    cv::Vec4d line;
    cv::fitLine(edge, line, cv::DIST_L2, 0, 0.01, 0.01);
    double k = line[1] / line[0], offset = line[3] - k * line[2];
    return { k, offset };
}

std::tuple<double, double> sfr::line_fit(const cv::Mat& mat) {
    cv::Mat img = mat.clone();
    cv::Mat derivate(img.size(), CV_64F), hammed(img.size(), CV_64F);
    cv::filter2D(img, derivate, CV_64F, sfr::kernel);
    std::vector<double> w = sfr::hamming(mat.cols);
    for (int i = 0; i < mat.rows; ++i)
        hammed.row(i) = derivate.row(i).mul(w);

    auto [k0, b0] = centroid_fit(hammed);

    for (int i = 0; i < mat.rows; ++i) {
        double place = k0 * i + b0;
        w = sfr::tukey(mat.cols, place);
        hammed.row(i) = derivate.row(i).mul(w);
    }
    return centroid_fit(hammed);
}

std::vector<double> sfr::esf(const cv::Mat& mat, const double k) {
    const int n = sfr::factor * mat.cols;
    const int delta = std::abs(std::round(sfr::factor * (mat.rows - 1) * - k));
    const int bwidth = n + delta + 150;
    std::vector<int> cnt(bwidth), acc(bwidth);
    std::vector<double> res(n);
    for (size_t i = 0; i < mat.rows; i++)
    {
        const uchar* ptr = mat.ptr<uchar>(i);
        for (size_t j = 0; j < mat.cols; j++)
        {
            int d = std::ceil(sfr::factor * (j - k * i));
            d = std::clamp(d, 0, bwidth);
            acc[d] += ptr[j]; ++cnt[d];
        }
    }
    cv::divide(acc, cnt, res);
    return std::vector(res.begin() + delta/2, res.begin() + delta/2 + n);
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
    // 插值计算
    double k = n * (*(point+1) - *point);
    return (0.1 - *point) / k + 1.0 * i / n;
}