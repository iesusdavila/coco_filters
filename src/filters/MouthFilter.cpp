#include "filters/MouthFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

MouthFilter::MouthFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat MouthFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 468) return frame;

    try {
        const int LEFT_MOUTH_CORNER = 61;
        const int RIGHT_MOUTH_CORNER = 291;
        const int UPPER_LIP = 13;
        const int LOWER_LIP = 14;

        cv::Point2f left_corner = landmarks[LEFT_MOUTH_CORNER];
        cv::Point2f right_corner = landmarks[RIGHT_MOUTH_CORNER];
        cv::Point2f upper_lip = landmarks[UPPER_LIP];
        cv::Point2f lower_lip = landmarks[LOWER_LIP];

        if (valid_landmark(left_corner) || valid_landmark(right_corner) ||
            valid_landmark(upper_lip) || valid_landmark(lower_lip)) {
            RCLCPP_ERROR(rclcpp::get_logger("MouthFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        float mouth_width = cv::norm(right_corner - left_corner);
        float mouth_height = cv::norm(lower_lip - upper_lip);
        

        cv::Mat mouth_asset = assets[current_asset_idx].clone();
        if (mouth_asset.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("MouthFilter"), "Asset sin canal alpha");
            return frame;
        }

        float aspect_ratio = static_cast<float>(mouth_asset.rows) / mouth_asset.cols;
        int target_width = static_cast<int>(mouth_width * 1.3f);
        int target_height = static_cast<int>(target_width * aspect_ratio);
        target_width = std::clamp(target_width, 20, 300);
        target_height = std::clamp(target_height, 10, 150);
        
        cv::resize(mouth_asset, mouth_asset, cv::Size(target_width, target_height));

        float dx = right_corner.x - left_corner.x;
        float dy = right_corner.y - left_corner.y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        cv::Mat rotated = rotate_image(mouth_asset, angle);
        if (rotated.empty()) return frame;

        int center_x = static_cast<int>((left_corner.x + right_corner.x) / 2 - rotated.cols / 2);
        int center_y = static_cast<int>((upper_lip.y + lower_lip.y) / 2 - rotated.rows / 2);

        optimized_overlay(frame, rotated, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("MouthFilter"), "Error: %s", e.what());
        return frame;
    }
}