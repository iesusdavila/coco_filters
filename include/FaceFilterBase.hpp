#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

class FaceFilter {
protected:
    std::vector<cv::Mat> assets;
    size_t current_asset_idx = 0;

public:
    size_t getCurrentIndex() const { return current_asset_idx; }
    void setCurrentIndex(size_t idx) { current_asset_idx = idx % assets.size(); }
    size_t getAssetsSize() const { return assets.size(); }
    void incrementIndex() { current_asset_idx = (current_asset_idx + 1) % assets.size(); }
    void decrementIndex() { current_asset_idx = (current_asset_idx - 1 + assets.size()) % assets.size(); }

    std::vector<cv::Mat> load_assets(const std::string& path);
    cv::Mat rotate_image(const cv::Mat& image, double angle);
    void optimized_overlay(cv::Mat& bg, const cv::Mat& overlay, int x, int y);

    FaceFilter() = default;
    explicit FaceFilter(const std::string& assets_path);
    virtual cv::Mat apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) = 0;
};