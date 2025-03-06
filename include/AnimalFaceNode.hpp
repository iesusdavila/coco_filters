#pragma once
#include "AnimalFilter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <buddy_interfaces/msg/face_landmarks.hpp>
#include <atomic>
#include <map>

using namespace std::chrono_literals;
using namespace message_filters;
using ImageMsg = sensor_msgs::msg::Image;
using FaceLandmarks = buddy_interfaces::msg::FaceLandmarks;
typedef sync_policies::ApproximateTime<ImageMsg, FaceLandmarks> ApproximateTimePolicy;

class AnimalFaceNode : public rclcpp::Node {
public:
    AnimalFaceNode();

    void shutdown() {
        running_ = false;
        if(keyboard_thread_.joinable()) keyboard_thread_.join();
    }
    
private:
    void callback(const ImageMsg::ConstSharedPtr& img_msg, 
                 const FaceLandmarks::ConstSharedPtr& landmarks_msg);
    void keyboard_listener();
    void change_filter(const std::string& animal);

    std::atomic<bool> running_{true};
    std::thread keyboard_thread_;
    std::mutex filter_mutex_;
    
    std::map<int, std::string> key_bindings_ = {
        {'1', "bear"},
        {'2', "cat"},
        {'3', "monkey"}
    };
    
    AnimalFilter current_filter_;
    image_transport::Publisher image_pub_;
    message_filters::Subscriber<ImageMsg> image_sub_;
    message_filters::Subscriber<FaceLandmarks> landmarks_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> sync_;
};