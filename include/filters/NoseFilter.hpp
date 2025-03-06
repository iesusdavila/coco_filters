#pragma once
#include "FaceFilterBase.hpp"
#include <opencv2/opencv.hpp>

class NoseFilter : public FaceFilter {
public:
    NoseFilter(const std::string& assets_path);
    cv::Mat apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) override;

private:
    bool is_visible(const cv::Point2f& landmark, const cv::Size& frame_size);
    bool validate_position(int x, int y, const cv::Size& asset_size, const cv::Size& frame_size);
};