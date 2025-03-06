#include "filters/NoseFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

NoseFilter::NoseFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat NoseFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 468) return frame;

    try {
        const int NOSE_TIP = 4;
        const int NOSE_BRIDGE = 6;
        const int LEFT_NOSTRIL = 98;
        const int RIGHT_NOSTRIL = 327;

        cv::Point2f nose_tip = landmarks[NOSE_TIP];
        cv::Point2f nose_bridge = landmarks[NOSE_BRIDGE];
        cv::Point2f left_nostril = landmarks[RIGHT_NOSTRIL];
        cv::Point2f right_nostril = landmarks[LEFT_NOSTRIL];

        if (std::isnan(nose_tip.x) || std::isinf(nose_tip.x) || 
            std::isnan(nose_bridge.y) || std::isinf(nose_bridge.y) ||
            std::isnan(left_nostril.x) || std::isinf(left_nostril.x) || 
            std::isnan(right_nostril.y) || std::isinf(right_nostril.y)) {
            RCLCPP_ERROR(rclcpp::get_logger("NoseFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        float nose_width = cv::norm(right_nostril - left_nostril);
        float vertical_distance = cv::norm(nose_bridge - nose_tip);
        int nose_height = static_cast<int>(vertical_distance * 1.2f);

        if (nose_width < 10 || nose_height < 10) return frame;

        cv::Mat nose_asset = assets[current_asset_idx].clone();
        if (nose_asset.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("NoseFilter"), "Asset sin canal alpha");
            return frame;
        }

        int target_width = static_cast<int>(nose_width * 1.5f);
        float aspect_ratio = static_cast<float>(nose_asset.rows) / nose_asset.cols;
        int target_height = static_cast<int>(target_width * aspect_ratio);
        
        target_width = std::clamp(target_width, 20, 200);
        target_height = std::clamp(target_height, 20, 200);
        
        cv::resize(nose_asset, nose_asset, cv::Size(target_width, target_height));

        float dx = right_nostril.x - left_nostril.x;
        float dy = right_nostril.y - left_nostril.y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;

        cv::Mat rotated = rotate_image(nose_asset, angle);
        if (rotated.empty()) return frame;

        int pos_x = static_cast<int>(nose_tip.x - rotated.cols / 2);
        int pos_y = static_cast<int>(nose_tip.y - rotated.rows / 2);

        if (!validate_position(pos_x, pos_y, rotated.size(), frame.size())) {
            return frame;
        }

        optimized_overlay(frame, rotated, pos_x, pos_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("NoseFilter"), "Error: %s", e.what());
        return frame;
    }
}

bool NoseFilter::is_visible(const cv::Point2f& landmark, const cv::Size& frame_size) {
    const float MARGIN = 0.1f;
    return (landmark.x > frame_size.width * MARGIN && 
            landmark.x < frame_size.width * (1 - MARGIN) &&
            landmark.y > frame_size.height * MARGIN && 
            landmark.y < frame_size.height * (1 - MARGIN));
}

bool NoseFilter::validate_position(int x, int y, const cv::Size& asset_size, const cv::Size& frame_size) {
    const float PADDING_FACTOR = 0.2f;
    return (x > -asset_size.width * PADDING_FACTOR) &&
           (y > -asset_size.height * PADDING_FACTOR) &&
           (x + asset_size.width < frame_size.width * (1 + PADDING_FACTOR)) &&
           (y + asset_size.height < frame_size.height * (1 + PADDING_FACTOR));
}