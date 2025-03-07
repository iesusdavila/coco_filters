#include "filters/FaceMaskFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

FaceMaskFilter::FaceMaskFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat FaceMaskFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 468) return frame;

    try {
        cv::Mat mask = assets[current_asset_idx].clone();
        if (current_asset_idx >= assets.size() || mask.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("FaceMaskFilter"), "Máscara sin canal alpha");
            return frame;
        }

        const int FACE_LEFT = 234;
        const int FACE_RIGHT = 454;

        cv::Point2f face_left = landmarks[FACE_LEFT];
        cv::Point2f face_right = landmarks[FACE_RIGHT];
        
        if (valid_landmark(face_left) || valid_landmark(face_right)) {
            RCLCPP_ERROR(rclcpp::get_logger("FaceFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        int left_x = static_cast<int>(face_left.x);
        int left_y = static_cast<int>(face_left.y);
        int right_x = static_cast<int>(face_right.x);
        int right_y = static_cast<int>(face_right.y);

        int dx = right_x - left_x;
        int dy = right_y - left_y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        float face_distance = std::hypot(dx, dy);

        if (face_distance < 10 || face_distance > 300) return frame;

        int face_width = static_cast<int>(face_distance * 1.8);
        face_width = std::clamp(face_width, 100, 1000);

        float aspect_ratio = static_cast<float>(mask.rows) / mask.cols;
        int face_height = static_cast<int>(face_width * aspect_ratio);
        face_height = std::clamp(face_height, 100, 1000);
        
        cv::Size target_size(face_width, face_height);
        if (target_size.width <= 0 || target_size.height <= 0) return frame;

        cv::resize(mask, mask, target_size);

        cv::Mat rotated = rotate_image(mask, angle);
        if (rotated.empty()) return frame;

        int center_x = (left_x + right_x) / 2 - rotated.cols / 2;
        int center_y = (landmarks[10].y + landmarks[152].y) / 2 - rotated.rows / 2;

        if (!validate_position(center_x, center_y, rotated.size(), frame.size())) {
            return frame;
        }

        optimized_overlay(frame, rotated, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("FaceMaskFilter"), "Error: %s", e.what());
        return frame;
    }
}