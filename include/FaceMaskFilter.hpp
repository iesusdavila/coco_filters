#pragma once
#include "FaceFilterBase.hpp"
#include <opencv2/opencv.hpp>

class FaceMaskFilter : public FaceFilter {
public:
    FaceMaskFilter(const std::string& assets_path);
    cv::Mat apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) override;

private:
    bool validate_position(int x, int y, const cv::Size& asset_size, const cv::Size& frame_size);
};