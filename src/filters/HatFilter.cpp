#include "filters/HatFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

HatFilter::HatFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat HatFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 333) return frame;

    try {
        const int FOREHEAD_RIGHT = 103;
        const int FOREHEAD_LEFT = 332; 

        cv::Point2f forehead_left = landmarks[FOREHEAD_LEFT];
        cv::Point2f forehead_right = landmarks[FOREHEAD_RIGHT];

        // Validación de landmarks
        if (std::isnan(forehead_left.x) || std::isinf(forehead_left.x) || 
            std::isnan(forehead_right.y) || std::isinf(forehead_right.y)) {
            RCLCPP_ERROR(rclcpp::get_logger("HatFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        // Cálculo de orientación y distancia
        float dx = forehead_right.x - forehead_left.x;
        float dy = forehead_right.y - forehead_left.y;
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        float forehead_distance = std::hypot(dx, dy);
        
        if (forehead_distance < 15.0f) return frame; // Distancia mínima

        // Preparación del asset
        cv::Mat hat_asset = assets[current_asset_idx].clone();
        if (hat_asset.channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("HatFilter"), "Asset sin canal alpha");
            return frame;
        }

        // Cálculo de tamaño
        int hat_width = static_cast<int>(forehead_distance * 1.7f);
        float aspect_ratio = static_cast<float>(hat_asset.rows) / hat_asset.cols;
        int hat_height = static_cast<int>(hat_width * aspect_ratio * 0.75f);
        
        hat_width = std::clamp(hat_width, 30, 500);
        hat_height = std::clamp(hat_height, 30, 500);
        
        cv::resize(hat_asset, hat_asset, cv::Size(hat_width, hat_height));

        // Rotación
        cv::Mat rotated_hat = rotate_image(hat_asset, angle);
        if (rotated_hat.empty()) return frame;

        // Posicionamiento
        int center_x = static_cast<int>((forehead_left.x + forehead_right.x)/2) - rotated_hat.cols/2;
        int center_y = static_cast<int>((forehead_left.y + forehead_right.y)/2) - rotated_hat.rows/2 - (hat_height/4);

        // Validación de posición
        if (!validate_position(center_x, center_y, rotated_hat.size(), frame.size())) {
            return frame;
        }

        // Aplicar overlay
        optimized_overlay(frame, rotated_hat, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("HatFilter"), "Error: %s", e.what());
        return frame;
    }
}

bool HatFilter::validate_position(int x, int y, const cv::Size& asset_size, const cv::Size& frame_size) {
    const float PADDING_FACTOR = 0.25f;
    return (x > -asset_size.width * PADDING_FACTOR) &&
           (y > -asset_size.height * PADDING_FACTOR) &&
           (x + asset_size.width < frame_size.width * (1 + PADDING_FACTOR)) &&
           (y + asset_size.height < frame_size.height * (1 + PADDING_FACTOR));
}