#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.hpp"
#include "cv_bridge/cv_bridge.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "buddy_interfaces/msg/face_landmarks.hpp"
#include "GlassesFilter.hpp"
#include "MouthFilter.hpp"
#include "NoseFilter.hpp"

using namespace std::chrono_literals;
using namespace message_filters;
using ImageMsg = sensor_msgs::msg::Image;
using FaceLandmarks = buddy_interfaces::msg::FaceLandmarks;
typedef sync_policies::ApproximateTime<ImageMsg, FaceLandmarks> ApproximateTimePolicy;

class FaceFilterNode : public rclcpp::Node {
public:
    FaceFilterNode() : Node("face_filter_node") {
        // Configurar QoS
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        
        // Inicializar filtro de gafas
        std::string assets_path = ament_index_cpp::get_package_share_directory("buddy-filters") + "/imgs";
        glasses_filter_ = std::make_shared<GlassesFilter>(assets_path + "/glasses");
        mouth_filter_ = std::make_shared<MouthFilter>(assets_path + "/mouths");
        nose_filter_ = std::make_shared<NoseFilter>(assets_path + "/noses");
        
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
        
        RCLCPP_INFO(this->get_logger(), "Nodo inicializado. Filtros activos: Gafas, Boca");
    }

private:
    void callback(const ImageMsg::ConstSharedPtr& img_msg, 
                  const FaceLandmarks::ConstSharedPtr& landmarks_msg) {
        try {
            // Convertir a OpenCV
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
            
            // Aplicar filtro de gafas
            if (!landmarks.empty()) {
                RCLCPP_INFO(this->get_logger(), "Aplicando filtro de gafas");
                frame = glasses_filter_->apply_filter(frame, landmarks, frame.size());
                RCLCPP_INFO(this->get_logger(), "Aplicando filtro de boca");
                frame = mouth_filter_->apply_filter(frame, landmarks, frame.size());
                RCLCPP_INFO(this->get_logger(), "Aplicando filtro de nariz");
                frame = nose_filter_->apply_filter(frame, landmarks, frame.size());
            }
            
            // Publicar imagen procesada
            auto output_msg = cv_bridge::CvImage(
                img_msg->header, 
                "bgr8", 
                frame
            ).toImageMsg();
            image_pub_.publish(output_msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error en callback: %s", e.what());
        }
    }

    std::shared_ptr<GlassesFilter> glasses_filter_;
    std::shared_ptr<MouthFilter> mouth_filter_;
    std::shared_ptr<NoseFilter> nose_filter_;
    image_transport::Publisher image_pub_;
    Subscriber<ImageMsg> image_sub_;
    Subscriber<FaceLandmarks> landmarks_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, 
            buddy_interfaces::msg::FaceLandmarks
        >
    >> sync_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FaceFilterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}