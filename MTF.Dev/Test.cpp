#include "Test.h"
#include "MTF.h"

void TestSlanted()
{
	cv::Mat img = cv::imread("../assets/slanted.tif", cv::IMREAD_GRAYSCALE);
	vd dists, grays, mtf;
	MTF::Slanted(img, dists, grays);
	MTF::MTF(dists, grays, mtf);

	auto min = std::min_element(mtf.begin(), mtf.end(), [](auto& l, auto& r) {return std::abs(l - 0.1) < std::abs(r - 0.1); });
	std::cout << "MTF10:" << std::distance(mtf.begin(), min) / 10.0  << std::endl;
}

void TestCylinder()
{
	cv::Mat img = cv::imread("../assets/cylinder.tif", cv::IMREAD_GRAYSCALE);
	vd dists, grays, mtf;
	MTF::Cylinder(img, dists, grays);
	MTF::MTF(dists, grays, mtf);

	auto min = std::min_element(mtf.begin(), mtf.end(), [](auto& l, auto& r) {return std::abs(l - 0.1) < std::abs(r - 0.1); });
	std::cout << "MTF10:" << std::distance(mtf.begin(), min) / 10.0 << std::endl;
}
