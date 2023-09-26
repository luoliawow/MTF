#include "Test.h"
#include <opencv2/core/utils/logger.hpp>

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	TestSlanted();

	TestCylinder();
}
