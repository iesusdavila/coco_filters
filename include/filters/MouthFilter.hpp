#pragma once
#include "FaceFilterBase.hpp"
#include <opencv2/opencv.hpp>

class MouthFilter : public FaceFilter {
public:
    MouthFilter(const std::string& assets_path);
    cv::Mat apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) override;
};