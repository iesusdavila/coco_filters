#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.hpp"
#include "cv_bridge/cv_bridge.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "buddy_interfaces/msg/face_landmarks.hpp"
#include "filters/GlassesFilter.hpp"
#include "filters/MouthFilter.hpp"
#include "filters/NoseFilter.hpp"
#include "filters/HatFilter.hpp"
#include "filters/FaceMaskFilter.hpp"
#include <atomic>
#include <thread>
#include <termios.h>
#include <unistd.h>

using namespace std::chrono_literals;
using namespace message_filters;
using ImageMsg = sensor_msgs::msg::Image;
using FaceLandmarks = buddy_interfaces::msg::FaceLandmarks;
typedef sync_policies::ApproximateTime<ImageMsg, FaceLandmarks> ApproximateTimePolicy;

class FaceFilterNode : public rclcpp::Node {
public:
    FaceFilterNode() : Node("face_filter_node"), 
                      mask_mode_(false),
                      running_(true) {
        // Configurar QoS
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        
        // Inicializar filtros
        std::string assets_path = ament_index_cpp::get_package_share_directory("buddy_filters") + "/imgs";
        glasses_filter_ = std::make_shared<GlassesFilter>(assets_path + "/glasses");
        mouth_filter_ = std::make_shared<MouthFilter>(assets_path + "/mouths");
        nose_filter_ = std::make_shared<NoseFilter>(assets_path + "/noses");
        hat_filter_ = std::make_shared<HatFilter>(assets_path + "/hats");
        face_mask_filter_ = std::make_shared<FaceMaskFilter>(assets_path + "/faces");
        
        // Configurar suscriptores
        image_sub_.subscribe(this, "image_raw");
        landmarks_sub_.subscribe(this, "face_landmarks");
        
        // Configurar sincronizador
        sync_ = std::make_shared<message_filters::Synchronizer<ApproximateTimePolicy>>(
            ApproximateTimePolicy(10), image_sub_, landmarks_sub_
        );
        sync_->registerCallback(
            std::bind(&FaceFilterNode::callback, 
            this, 
            std::placeholders::_1, 
            std::placeholders::_2)
        );
        
        // Configurar publisher
        image_pub_ = image_transport::create_publisher(this, "filtered_image", qos.get_rmw_qos_profile());
        
        // Hilo para entrada de teclado
        keyboard_thread_ = std::thread(&FaceFilterNode::keyboardListener, this);
        
        RCLCPP_INFO(this->get_logger(), "Nodo inicializado. Modo normal activado");
    }

    ~FaceFilterNode() {
        running_ = false;
        if(keyboard_thread_.joinable()) keyboard_thread_.join();
    }

private:
    void callback(const ImageMsg::ConstSharedPtr& img_msg, 
                  const FaceLandmarks::ConstSharedPtr& landmarks_msg) {
        try {
            cv::Mat yuv_image(img_msg->height, img_msg->width, CV_8UC2, const_cast<uchar*>(img_msg->data.data()));
            cv::Mat frame;
            cv::cvtColor(yuv_image, frame, cv::COLOR_YUV2BGR_YUYV);
            
            // Convertir landmarks
            std::vector<cv::Point2f> landmarks;
            for (const auto& point : landmarks_msg->landmarks) {
                landmarks.emplace_back(
                    (1.0 - point.x) * frame.cols,
                    point.y * frame.rows
                );
            }
            
            // Aplicar filtros según modo
            if (!landmarks.empty()) {
                if(mask_mode_) {
                    frame = face_mask_filter_->apply_filter(frame, landmarks, frame.size());
                } else {
                    frame = glasses_filter_->apply_filter(frame, landmarks, frame.size());
                    frame = mouth_filter_->apply_filter(frame, landmarks, frame.size());
                    frame = nose_filter_->apply_filter(frame, landmarks, frame.size());
                    frame = hat_filter_->apply_filter(frame, landmarks, frame.size());
                }
            }
            
            // Publicar imagen procesada
            auto output_msg = cv_bridge::CvImage(img_msg->header, "bgr8", frame).toImageMsg();
            image_pub_.publish(output_msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error en callback: %s", e.what());
        }
    }

    void keyboardListener() {
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        while (running_) {
            char key = getchar();
            std::lock_guard<std::mutex> lock(mutex_);
            
            if(mask_mode_) {
                handleMaskMode(key);
            } else {
                handleNormalMode(key);
            }
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }

    void handleNormalMode(char key) {
        switch(key) {
            case 'e': incrementIndex(hat_filter_); break;
            case 'q': decrementIndex(hat_filter_); break;
            case 'w': incrementIndex(nose_filter_); break;
            case 's': decrementIndex(nose_filter_); break;
            case 'd': incrementIndex(glasses_filter_); break;
            case 'a': decrementIndex(glasses_filter_); break;
            case 'c': incrementIndex(mouth_filter_); break;
            case 'z': decrementIndex(mouth_filter_); break;
            case 'n': case 'm': 
                mask_mode_ = true;
                RCLCPP_INFO(get_logger(), "Modo máscara activado");
                break;
        }
    }

    void handleMaskMode(char key) {
        switch(key) {
            case 'n': decrementIndex(face_mask_filter_); break;
            case 'm': incrementIndex(face_mask_filter_); break;
            default: 
                mask_mode_ = false;
                RCLCPP_INFO(get_logger(), "Modo normal activado");
                break;
        }
    }

    template<typename T>
    void incrementIndex(std::shared_ptr<T> filter) {
        if(filter->getAssetsSize() > 0) {
            filter->incrementIndex();
            RCLCPP_INFO(get_logger(), "Índice actualizado: %zu", filter->getCurrentIndex());
        }
    }

    template<typename T>
    void decrementIndex(std::shared_ptr<T> filter) {
        if(filter->getAssetsSize() > 0) {
            filter->decrementIndex();
            RCLCPP_INFO(get_logger(), "Índice actualizado: %zu", filter->getCurrentIndex());
        }
    }

    // Variables de estado
    std::atomic<bool> mask_mode_;
    std::atomic<bool> running_;
    std::thread keyboard_thread_;
    std::mutex mutex_;

    // Filtros
    std::shared_ptr<GlassesFilter> glasses_filter_;
    std::shared_ptr<MouthFilter> mouth_filter_;
    std::shared_ptr<NoseFilter> nose_filter_;
    std::shared_ptr<HatFilter> hat_filter_;
    std::shared_ptr<FaceMaskFilter> face_mask_filter_;
    
    image_transport::Publisher image_pub_;
    Subscriber<ImageMsg> image_sub_;
    Subscriber<FaceLandmarks> landmarks_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> sync_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FaceFilterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}