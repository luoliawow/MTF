#include <sfr/general.h>
#include <Eigen/Dense>

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

std::vector<double> sfr::polyfit(const std::vector<double> &x, const std::vector<double> &y, int degree) {
    assert(x.size() == y.size());
    // polyfit
    Eigen::VectorXd Y = Eigen::Map<const Eigen::VectorXd>(y.data(), y.size());
    Eigen::MatrixXd A(x.size(), degree + 1);
    for (int i = 0; i < x.size(); i++)
        for (int j = 0; j <= degree; j++)
            A(i, j) = pow(x[i], j);
    Eigen::VectorXd coef = A.householderQr().solve(Y);
    return std::vector<double>(coef.data(), coef.data() + coef.size());
}

std::vector<double> sfr::polyval(const std::vector<double> &x, const std::vector<double> &coeff) {
    std::vector<double> out(x.size(), 0);
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < coeff.size(); j++)
            out[i] += coeff[j] * pow(x[i], j);
    }
    return out;
}
