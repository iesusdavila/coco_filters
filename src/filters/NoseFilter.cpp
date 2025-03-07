#include "filters/NoseFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

NoseFilter::NoseFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat NoseFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 468) return frame;

    try {
        cv::Mat nose = assets[current_asset_idx].clone();
        if (current_asset_idx >= assets.size() || nose.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("NoseFilter"), "Asset sin canal alpha");
            return frame;
        }

        const int LEFT_NOSTRIL = 327;
        const int RIGHT_NOSTRIL = 98;

        cv::Point2f left_nostril = landmarks[RIGHT_NOSTRIL];
        cv::Point2f right_nostril = landmarks[LEFT_NOSTRIL];

        if (valid_landmark(left_nostril) || valid_landmark(right_nostril)) {
            RCLCPP_ERROR(rclcpp::get_logger("NoseFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        int left_x = static_cast<int>(left_nostril.x);
        int left_y = static_cast<int>(left_nostril.y);
        int right_x = static_cast<int>(right_nostril.x);
        int right_y = static_cast<int>(right_nostril.y);

        int dx = right_x - left_x;
        int dy = right_y - left_y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        float nose_distance = std::hypot(dx, dy);

        if (nose_distance < 10 || nose_distance > 300) return frame;

        int nose_width = static_cast<int>(nose_distance * 1.5);
        nose_width = std::clamp(nose_width, 20, 200);

        float aspect_ratio = static_cast<float>(nose.rows) / nose.cols;
        int nose_height = static_cast<int>(nose_width * aspect_ratio);
        nose_height = std::clamp(nose_height, 20, 200);

        cv::Size target_size(nose_width, nose_height);
        if (target_size.width <= 0 || target_size.height <= 0) return frame;
        
        cv::resize(nose, nose, target_size);

        cv::Mat rotated = rotate_image(nose, angle);
        if (rotated.empty()) return frame;

        int pos_x = landmarks[5].x - rotated.cols / 2;
        int pos_y = landmarks[5].y - rotated.rows / 2;

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