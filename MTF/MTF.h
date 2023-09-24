#pragma once

#include <ranges>
#include <vector>
#include <opencv2/opencv.hpp>

using vd = std::vector<double>;
namespace rv = std::ranges::views;
namespace ra = std::ranges;

namespace MTF
{
    static void MTF(const vd& dists, const vd& grays, vd& mtf);
    static vd Linspace(const double begin, const double end, const size_t n);
    static void Slanted(const cv::Mat& img, vd& dists, vd& grays, float thresh = 5.0f);
    static void Cylinder(const cv::Mat& img, vd& dists, vd& grays, float thresh = 15.0f);
}
