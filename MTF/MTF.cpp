#include "MTF.h"

#include <gsl/gsl_spline.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_multifit.h>

vd BSpline(const vd& x, const vd& y, const vd& x_est)
{
    assert(x.size() == y.size());
    const size_t n = x.size(), k = 4, ncoeff = 26;
    
    gsl_bspline_workspace* bw = gsl_bspline_alloc(k, ncoeff - 2);
    gsl_bspline_knots_uniform(x.front(), x.back(), bw);
    
    gsl_vector *c = gsl_vector_alloc(ncoeff), *B = gsl_vector_alloc(ncoeff);
    gsl_vector *vx = gsl_vector_alloc(n), *vy = gsl_vector_alloc(n);
    for (int i = 0; i < n; ++i)
    {
        gsl_vector_set(vx, i, x[i]); gsl_vector_set(vy, i, y[i]);
    }

    gsl_matrix *X = gsl_matrix_alloc(n, ncoeff), *cov = gsl_matrix_alloc(ncoeff, ncoeff);
    gsl_multifit_linear_workspace *mw = gsl_multifit_linear_alloc(n, ncoeff);
    for (int i = 0; i < n; ++i)
    {
        double xi = gsl_vector_get(vx, i);
        gsl_bspline_eval(xi, B, bw);
        for (int j = 0; j < ncoeff; ++j)
        {
            gsl_matrix_set(X, i, j, gsl_vector_get(B, j));
        }
    }

    vd result; double chisq;
    gsl_multifit_linear(X, vy, c, cov, &chisq, mw);
    for (double xi : x_est)
    {
        double yi, yerr;
        gsl_bspline_eval(xi, B, bw);
        gsl_multifit_linear_est(B, c, cov, &yi, &yerr);
        result.emplace_back(yi);
    }

    gsl_vector_free(vy); gsl_vector_free(vx);
    gsl_vector_free(B); gsl_vector_free(c);
    gsl_matrix_free(X); gsl_matrix_free(cov);
    gsl_bspline_free(bw); gsl_multifit_linear_free(mw);

    return result;
}

vd CSpline(const vd& x, const vd& y, const vd& x_est)
{
    assert(x.size() == y.size());
    const size_t n = x.size();
    
    gsl_interp_accel* acc = gsl_interp_accel_alloc();
    gsl_spline* interp = gsl_spline_alloc(gsl_interp_cspline, n);
    gsl_spline_init(interp, x.data(), y.data(), n);

    vd result;
    for (double xi : x_est)
    {
        double yi = gsl_spline_eval(interp, xi, acc);
        result.emplace_back(yi);
    }

    gsl_interp_accel_free(acc);
    gsl_spline_free(interp);

    return result;
}

void ESF_Interp(const std::vector<std::tuple<double, double>>& dvps, const int ninterp, vd& dists, vd& grays)
{
    dists = rv::iota(0, ninterp)
        | rv::transform([&](int i) {return std::lerp(std::get<0>(dvps.front()), std::get<0>(dvps.back()), 1.0 / ninterp * i); })
        | ra::to<std::vector>();
    vd dist_t = dvps | rv::elements<0> | ra::to<std::vector>();
    vd gray_t = dvps | rv::elements<1> | ra::to<std::vector>();
    grays = BSpline(dist_t, gray_t, dists);
}


// return [slope, offset] from the edge line
std::tuple<float, float> EdgeFit(const cv::Mat& roi, std::vector<cv::Point>& points)
{
    cv::Mat edge;
    cv::Canny(roi, edge, 40, 90, 3, true);
    cv::findNonZero(edge, points);
    
    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    float slope = line[1] / line[0], offset = line[3] - slope * line[2];
    return std::tuple { slope, offset};
}

// Input dist & grays should be the same size
// dist should be sorted in ascending order
// Assume dist & grays have been filtered with dist_thresh
void MTF::MTF(const vd& dists, const vd& grays, vd& mtf)
{
    assert(dists.size() == grays.size());
    constexpr int window = 3;
    
    auto diff = rv::zip(dists, grays) 
        | rv::adjacent_transform<window>([](auto l, auto, auto r)
        {
            double x1 = std::get<0>(l), y1 = std::get<1>(l);
            double x2 = std::get<0>(r), y2 = std::get<1>(r);
            return (y2 - y1) / (x2 - x1);
        });
    double max = ra::max(diff);
    auto lsf = diff
        | rv::transform([max](double x) { return x / max; })
        | ra::to<std::vector>();

    const double sum = ra::fold_left(lsf, 0.0, std::plus<>());
    std::vector<std::complex<double>> fft(lsf.size());
    cv::dft(lsf,fft,cv::DFT_COMPLEX_OUTPUT);
    mtf = fft
        | rv::transform([&](auto& c) { return std::abs(c) / sum; })
        | ra::to<std::vector>();

    // TODO: Add frequency axis
    
    // TODO: Interpolate the MTF
}

vd MTF::Linspace(const double begin, const double end, const size_t n)
{
    vd result;
    for (int i = 0; i < n; ++i)
        result.emplace_back(std::lerp(begin, end, 1.0 / n * i));
    return result;
}

vd Hamming(const int n, const int mid) {
    vd hammingVector(n);
    const int radius = std::max(mid - 1, n - mid);
    for (int i = 0; i < n; i++)
        hammingVector[i] = 0.54 + 0.46 * cos(CV_PI * (i - mid) / radius);
    return hammingVector;
}

// apply hamming to each row of roi
void ApplyHamming(const cv::Mat& roi, const std::vector<cv::Point>& edgePoints, cv::Mat& hammed)
{
    roi.convertTo(hammed, CV_32F);
    for (int i = 0; i < roi.rows; ++i)
    {
        vd hamvec = Hamming(roi.rows, edgePoints[i].x);
        float* row = hammed.ptr<float>(i);
        for (int j = 0; j < roi.cols; ++j) row[j] *= hamvec[j];
    }
}

void MTF::Slanted(const cv::Mat& img, vd& dists, vd& grays, float thresh)
{
    // get the edge and estimate its slope & offset
    std::vector<cv::Point> points;
    auto [slope, offset] = EdgeFit(img, points);
    
    cv::Mat hammed;
    ApplyHamming(img, points, hammed);

    std::vector<std::tuple<double, double>> dvps;
    const float upper = thresh, lower = -thresh;
    for (int i = 0; i < hammed.total(); ++i)
    {
        int x = i % hammed.cols, y = i / hammed.cols;
        float distance = (y - slope * x - offset) / std::sqrt(slope * slope + 1);
        if (distance >= lower && distance <= upper)
            dvps.emplace_back(distance, hammed.at<float>(y, x));
    }
    ra::sort(dvps, std::less<>(), [](auto& dvp) { return std::get<0>(dvp); });

    // interpolate the dist-gray pairs to 1000 points
    const int ninterp = 1000;
    ESF_Interp(dvps, ninterp, dists, grays);
}

// assume img only contains one circle
void MTF::Cylinder(const cv::Mat& img, vd& dists, vd& grays, float thresh)
{
    cv::Mat blur;
    cv::GaussianBlur(img, blur, cv::Size(9, 9), 2, 4);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blur, circles, cv::HOUGH_GRADIENT, 1, img.rows / 8.0, 70, 30, 100, 1000);
    float cx = circles[0][0], cy = circles[0][1], r = circles[0][2];

    std::vector<std::tuple<double, double>> dvps;
    const float upper = r + thresh, lower = r - thresh;
    for (int i = 0; i < blur.total(); i++)
    {
        int x = i % blur.cols, y = i / blur.cols;
        float dist = std::hypot(x - cx, y - cy);
        if (dist >= lower && dist <= upper)
            dvps.emplace_back(dist, blur.at<uchar>(y, x));
    }
    ra::sort(dvps, std::less<>{}, [](auto& dvp) {return std::get<0>(dvp); });

    const int ninterp = 1000;
    ESF_Interp(dvps, ninterp, dists, grays);
}

