#include "FaceFilterBase.hpp"
#include "rclcpp/rclcpp.hpp"
#include <iostream>

using namespace cv;
namespace fs = std::filesystem;

std::vector<cv::Mat> FaceFilter::load_assets(const std::string& path) {
    std::vector<cv::Mat> loaded_assets;
    for (const auto& entry : fs::directory_iterator(path)) {
        Mat img = imread(entry.path().string(), IMREAD_UNCHANGED);
        if (!img.empty() && img.cols > 0 && img.rows > 0) {
            loaded_assets.push_back(img);
        }
    }
    return loaded_assets;
}

FaceFilter::FaceFilter(const std::string& assets_path) {
    this->assets = load_assets(assets_path);
}

cv::Mat FaceFilter::rotate_image(const cv::Mat& image, double angle) {
    if (image.empty() || angle > 360.0 || angle < -360.0) return cv::Mat();
    
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    
    // --- Validación de tamaño de rotación ---
    double sin = std::abs(rot_mat.at<double>(0, 1));
    double cos = std::abs(rot_mat.at<double>(0, 0));
    int new_w = static_cast<int>(image.rows * sin + image.cols * cos);
    int new_h = static_cast<int>(image.rows * cos + image.cols * sin);
    
    if (new_w > 1000 || new_h > 1000) return cv::Mat(); // Límite absoluto
    
    rot_mat.at<double>(0, 2) += (new_w - image.cols) / 2.0;
    rot_mat.at<double>(1, 2) += (new_h - image.rows) / 2.0;
    
    cv::Mat rotated;
    cv::warpAffine(image, rotated, rot_mat, cv::Size(new_w, new_h));
    return rotated;
}

void FaceFilter::optimized_overlay(cv::Mat& bg, const cv::Mat& overlay, int x, int y) {
    // Verificaciones iniciales
    if (overlay.empty() || bg.empty() ||
        x >= bg.cols || y >= bg.rows ||
        x + overlay.cols <= 0 || y + overlay.rows <= 0) {
        return;
    }

    // Región de interés
    int y1 = std::max(y, 0);
    int y2 = std::min(y + overlay.rows, bg.rows);
    int x1 = std::max(x, 0);
    int x2 = std::min(x + overlay.cols, bg.cols);

    if (x1 >= x2 || y1 >= y2) return;

    // Asegurar que el fondo sea de 3 canales
    cv::Mat bg_3channel;
    if (bg.channels() == 4) {
        cv::cvtColor(bg, bg_3channel, cv::COLOR_BGRA2BGR);
    } else if (bg.channels() == 3) {
        bg_3channel = bg;
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("FaceFilter"), "Unsupported background image channels");
        return;
    }

    // Región de interés del fondo
    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat bg_roi = bg_3channel(roi);

    // Región de interés de la superposición
    cv::Rect overlay_roi(x1 - x, y1 - y, x2 - x1, y2 - y1);
    cv::Mat overlay_region = overlay(overlay_roi);

    // Separar canales
    std::vector<cv::Mat> overlay_channels(4);
    cv::split(overlay_region, overlay_channels);

    // Canal alpha normalizado
    cv::Mat alpha;
    overlay_channels[3].convertTo(alpha, CV_32F, 1.0/255.0);
    cv::Mat inv_alpha = 1.0 - alpha;

    // Canales de color de la superposición
    std::vector<cv::Mat> overlay_color_channels(3);
    for (int i = 0; i < 3; ++i) {
        overlay_color_channels[i] = overlay_channels[i].clone();
    }

    // Canales de color del fondo
    std::vector<cv::Mat> bg_channels(3);
    cv::split(bg_roi, bg_channels);

    // Blending
    for (int c = 0; c < 3; ++c) {
        cv::Mat bg_chan_float, ovl_chan_float;
        bg_channels[c].convertTo(bg_chan_float, CV_32F);
        overlay_color_channels[c].convertTo(ovl_chan_float, CV_32F);

        cv::Mat blended = bg_chan_float.mul(inv_alpha) + ovl_chan_float.mul(alpha);
        blended.convertTo(bg_channels[c], CV_8U);
    }

    // Fusionar canales
    cv::merge(bg_channels, bg_roi);
}