#ifndef GENERAL_H
#define GENERAL_H
#include <vector>

namespace sfr {
    std::vector<double> tukey(const int n0, const double mid);
    inline std::vector<double> hamming(const int n) { return tukey(n, n/2); }
    std::vector<double> center_shift(const std::vector<double>& x, const int center);
    std::vector<double> lsf(const std::vector<double>& esf);
    std::vector<double> mtf(const std::vector<double>& lsf);
}


#endif //GENERAL_H
