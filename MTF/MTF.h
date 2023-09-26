#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

using vd = std::vector<double>;

#ifdef MTF_EXPORTS
#define MTF_API __declspec(dllexport)
#else
#define MTF_API __declspec(dllimport)
#endif

namespace MTF
{
    MTF_API void MTF(const vd& dists, const vd& grays, vd& mtf);
    MTF_API vd Linspace(const double begin, const double end, const size_t n);
    MTF_API void Slanted(const cv::Mat& img, vd& dists, vd& grays, float thresh = 5.0f);
    MTF_API void Cylinder(const cv::Mat& img, vd& dists, vd& grays, float thresh = 15.0f);
}
