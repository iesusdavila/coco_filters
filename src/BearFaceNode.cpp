#include "AnimalFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <buddy_interfaces/msg/face_landmarks.hpp>

using namespace std::chrono_literals;
using namespace message_filters;
using ImageMsg = sensor_msgs::msg::Image;
using FaceLandmarks = buddy_interfaces::msg::FaceLandmarks;
typedef sync_policies::ApproximateTime<ImageMsg, FaceLandmarks> ApproximateTimePolicy;

class BearFaceNode : public rclcpp::Node {
public:
    BearFaceNode() : Node("bear_face_node"), filter_("bear") {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        
        image_sub_.subscribe(this, "image_raw");
        landmarks_sub_.subscribe(this, "face_landmarks");
        
        sync_ = std::make_shared<Synchronizer<ApproximateTimePolicy>>(
            ApproximateTimePolicy(10), image_sub_, landmarks_sub_);
        
        sync_->registerCallback(&BearFaceNode::callback, this);
        
        image_pub_ = image_transport::create_publisher(this, "filtered_image", qos.get_rmw_qos_profile());
        
        RCLCPP_INFO(get_logger(), "Nodo de filtro de oso inicializado");
    }

private:
    void callback(const ImageMsg::ConstSharedPtr& img_msg, 
                 const FaceLandmarks::ConstSharedPtr& landmarks_msg) {
        try {
            RCLCPP_INFO(get_logger(), "Recibiendo imagen y landmarks");
            cv::Mat yuv_image(img_msg->height, img_msg->width, CV_8UC2, const_cast<uchar*>(img_msg->data.data()));
            cv::Mat frame;
            cv::cvtColor(yuv_image, frame, cv::COLOR_YUV2BGR_YUYV);
            
            std::vector<cv::Point2f> landmarks;
            
            for(const auto& point : landmarks_msg->landmarks) {
                landmarks.emplace_back(
                    point.x * frame.cols,  // Reflejar en eje X
                    point.y * frame.rows
                );
            }
            
            cv::flip(frame, frame, 1);
            
            frame = filter_.apply_filter(frame, landmarks);
            
            auto output_msg = cv_bridge::CvImage(
                img_msg->header, "bgr8", frame).toImageMsg();
            image_pub_.publish(output_msg);
            
        } catch(const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error en callback: %s", e.what());
        }
    }

    AnimalFilter filter_;
    image_transport::Publisher image_pub_;
    message_filters::Subscriber<ImageMsg> image_sub_;
    message_filters::Subscriber<FaceLandmarks> landmarks_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> sync_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BearFaceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}