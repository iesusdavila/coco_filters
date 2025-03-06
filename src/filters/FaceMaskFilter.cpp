#include "filters/FaceMaskFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

FaceMaskFilter::FaceMaskFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat FaceMaskFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 455) return frame;

    try {
        const int FOREHEAD_TOP = 10;
        const int CHIN_BOTTOM = 152;
        const int FACE_LEFT = 454;
        const int FACE_RIGHT = 234;

        cv::Point2f forehead = landmarks[FOREHEAD_TOP];
        cv::Point2f chin = landmarks[CHIN_BOTTOM];
        cv::Point2f face_left = landmarks[FACE_LEFT];
        cv::Point2f face_right = landmarks[FACE_RIGHT];
        
        if (valid_landmark(forehead) || valid_landmark(chin) || 
            valid_landmark(face_left) || valid_landmark(face_right)) {
            RCLCPP_ERROR(rclcpp::get_logger("FaceFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        float face_width = cv::norm(face_right - face_left);
        float face_height = cv::norm(chin - forehead);
        
        if (face_width < 50.0f || face_height < 50.0f) return frame;

        cv::Mat mask = assets[current_asset_idx].clone();
        if (mask.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("FaceMaskFilter"), "Máscara sin canal alpha");
            return frame;
        }

        float aspect_ratio = static_cast<float>(mask.rows) / mask.cols;
        int target_width = static_cast<int>(face_width * 1.8f);
        int target_height = static_cast<int>(target_width * aspect_ratio);
        
        target_width = std::clamp(target_width, 100, 1000);
        target_height = std::clamp(target_height, 100, 1000);
        
        cv::resize(mask, mask, cv::Size(target_width, target_height));

        float dx = face_right.x - face_left.x;
        float dy = face_right.y - face_left.y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;

        cv::Mat rotated_mask = rotate_image(mask, angle);
        if (rotated_mask.empty()) return frame;

        int center_x = static_cast<int>((face_left.x + face_right.x)/2) - rotated_mask.cols/2;
        int center_y = static_cast<int>((forehead.y + chin.y)/2) - rotated_mask.rows/2;

        if (!validate_position(center_x, center_y, rotated_mask.size(), frame.size())) {
            return frame;
        }

        optimized_overlay(frame, rotated_mask, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("FaceMaskFilter"), "Error: %s", e.what());
        return frame;
    }
}

bool FaceMaskFilter::validate_position(int x, int y, const cv::Size& asset_size, const cv::Size& frame_size) {
    const float PADDING_FACTOR = 0.3f;
    return (x > -asset_size.width * PADDING_FACTOR) &&
           (y > -asset_size.height * PADDING_FACTOR) &&
           (x + asset_size.width < frame_size.width * (1 + PADDING_FACTOR)) &&
           (y + asset_size.height < frame_size.height * (1 + PADDING_FACTOR)) &&
           (asset_size.width > 50) &&  
           (asset_size.height > 50);
}