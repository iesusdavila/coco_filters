#include "filters/HatFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

HatFilter::HatFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat HatFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 468) return frame;

    try {
        cv::Mat hat = assets[current_asset_idx].clone();
        if (current_asset_idx >= assets.size() || hat.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("HatFilter"), "Asset sin canal alpha");
            return frame;
        }

        const int FOREHEAD_RIGHT = 332;
        const int FOREHEAD_LEFT = 103; 

        cv::Point2f forehead_left = landmarks[FOREHEAD_LEFT];
        cv::Point2f forehead_right = landmarks[FOREHEAD_RIGHT];

        if (valid_landmark(forehead_left) || valid_landmark(forehead_right)) {
            RCLCPP_ERROR(rclcpp::get_logger("HatFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        int left_x = static_cast<int>(forehead_left.x);
        int left_y = static_cast<int>(forehead_left.y);
        int right_x = static_cast<int>(forehead_right.x);
        int right_y = static_cast<int>(forehead_right.y);

        float dx = right_x - left_x;
        float dy = right_y - left_y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        float forehead_distance = std::hypot(dx, dy);
        
        if (forehead_distance < 15.0f) return frame;

        int hat_width = static_cast<int>(forehead_distance * 1.7f);
        hat_width = std::clamp(hat_width, 30, 500);

        float aspect_ratio = static_cast<float>(hat.rows) / hat.cols;
        int hat_height = static_cast<int>(hat_width * aspect_ratio * 0.75f); 
        hat_height = std::clamp(hat_height, 30, 500);
        
        cv::Size target_size(hat_width, hat_height);
        if (target_size.width <= 0 || target_size.height <= 0) return frame;

        cv::resize(hat, hat, target_size);

        cv::Mat rotated = rotate_image(hat, angle);
        if (rotated.empty()) return frame;

        int center_x = (left_x + right_x) / 2 - rotated.cols/2;
        int center_y = (left_y + right_y) / 2 - rotated.rows/2 - (hat_height/4);

        if (!validate_position(center_x, center_y, rotated.size(), frame.size())) {
            return frame;
        }

        optimized_overlay(frame, rotated, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("HatFilter"), "Error: %s", e.what());
        return frame;
    }
}