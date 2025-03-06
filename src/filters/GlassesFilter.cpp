#include "filters/GlassesFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

GlassesFilter::GlassesFilter(const std::string& assets_path) : FaceFilter(assets_path) {}

cv::Mat GlassesFilter::apply_filter(cv::Mat frame, const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_size) {
    if (frame.empty() || assets.empty() || landmarks.size() < 264) return frame;

    try {
        // --- 1. Validación extrema de landmarks ---
        cv::Point2f left_eye = landmarks[263];
        cv::Point2f right_eye = landmarks[33];
        if (std::isnan(left_eye.x) || std::isinf(left_eye.x) || 
            std::isnan(left_eye.y) || std::isinf(left_eye.y) ||
            std::isnan(right_eye.x) || std::isinf(right_eye.x) || 
            std::isnan(right_eye.y) || std::isinf(right_eye.y)) {
            RCLCPP_ERROR(rclcpp::get_logger("GlassesFilter"), "Landmarks inválidos (NaN/Inf)");
            return frame;
        }

        // --- 2. Conversión explícita a enteros ---
        int left_x = static_cast<int>(left_eye.x);
        int left_y = static_cast<int>(left_eye.y);
        int right_x = static_cast<int>(right_eye.x);
        int right_y = static_cast<int>(right_eye.y);

        // --- 3. Cálculo seguro de distancia ---
        int dx = right_x - left_x;
        int dy = right_y - left_y;
        double eye_distance = std::hypot(dx, dy);
        if (eye_distance < 10 || eye_distance > 300) return frame;

        // --- 4. Tamaño con redondeo explícito ---
        int glasses_width = static_cast<int>(std::round(eye_distance * 1.2));
        glasses_width = std::clamp(glasses_width, 20, 200);

        // --- 5. Validación de asset (canal alpha) ---
        if (current_asset_idx >= assets.size() || assets[current_asset_idx].channels() != 4) {
            RCLCPP_ERROR(rclcpp::get_logger("GlassesFilter"), "Asset no tiene canal alpha");
            return frame;
        }
        cv::Mat glasses = assets[current_asset_idx].clone();

        // --- 6. Redondeo y conversión a Size ---
        cv::Size target_size(
            std::max(1, glasses_width), 
            std::max(1, static_cast<int>(std::round(glasses.rows * glasses_width / static_cast<double>(glasses.cols))))
        );
        if (target_size.width <= 0 || target_size.height <= 0) return frame;

        cv::resize(glasses, glasses, target_size);

        // --- 7. Rotación con ángulo limitado ---
        double angle = -std::atan2(dy, dx) * 180.0 / CV_PI;
        angle = std::fmod(angle, 360.0); // Limitar a ±360°
        cv::Mat rotated = rotate_image(glasses, angle);
        if (rotated.empty() || rotated.cols > 1000 || rotated.rows > 1000) return frame;

        // --- 8. Posicionamiento con overflow check ---
        int center_x = (left_x + right_x) / 2 - rotated.cols / 2;
        int center_y = (left_y + right_y) / 2 - rotated.rows / 2;
        if (center_x < -rotated.cols || center_x > frame.cols + rotated.cols ||
            center_y < -rotated.rows || center_y > frame.rows + rotated.rows) {
            return frame;
        }

        optimized_overlay(frame, rotated, center_x, center_y);
        return frame;

    } catch (const cv::Exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("GlassesFilter"), "Error crítico: %s", e.what());
        return frame;
    }
}