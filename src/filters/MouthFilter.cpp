#include "filters/MouthFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

MouthFilter::MouthFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat MouthFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 468) return frame;

    try {
        cv::Mat mouth = assets[current_asset_idx].clone();
        if (mouth.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("MouthFilter"), "Asset sin canal alpha");
            return frame;
        }

        const int LEFT_MOUTH_CORNER = 61;
        const int RIGHT_MOUTH_CORNER = 291;

        cv::Point2f left_corner = landmarks[LEFT_MOUTH_CORNER];
        cv::Point2f right_corner = landmarks[RIGHT_MOUTH_CORNER];

        if (valid_landmark(left_corner) || valid_landmark(right_corner)) {
            RCLCPP_ERROR(rclcpp::get_logger("MouthFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        int left_x = static_cast<int>(left_corner.x);
        int left_y = static_cast<int>(left_corner.y);
        int right_x = static_cast<int>(right_corner.x);
        int right_y = static_cast<int>(right_corner.y);

        float dx = right_x - left_x;
        float dy = right_y - left_y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        float mouth_distance = std::hypot(dx, dy);

        if (mouth_distance < 10 || mouth_distance > 300) return frame;

        int mouth_width = static_cast<int>(mouth_distance * 1.3);
        mouth_width = std::clamp(mouth_width, 20, 300);

        float aspect_ratio = static_cast<float>(mouth.rows) / mouth.cols;
        int mouth_height = static_cast<int>(mouth_width * aspect_ratio);
        mouth_height = std::clamp(mouth_height, 10, 150);
        
        cv::Size target_size(mouth_width, mouth_height);
        if (target_size.width <= 0 || target_size.height <= 0) return frame;

        cv::resize(mouth, mouth, target_size);

        cv::Mat rotated = rotate_image(mouth, angle);
        if (rotated.empty()) return frame;

        int center_x = (left_x + right_x) / 2 - rotated.cols / 2;
        int center_y = (landmarks[13].y + landmarks[14].y) / 2 - rotated.rows / 2;

        if (!validate_position(center_x, center_y, rotated.size(), frame.size())) {
            return frame;
        }

        optimized_overlay(frame, rotated, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("MouthFilter"), "Error: %s", e.what());
        return frame;
    }
}