#include "AnimalFaceNode.hpp"
#include <termios.h>
#include <unistd.h>

AnimalFaceNode::AnimalFaceNode() : Node("animal_face_node"), current_filter_("bear") {
    // Configuración de QoS
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    
    // Suscriptores
    image_sub_.subscribe(this, "image_raw");
    landmarks_sub_.subscribe(this, "face_landmarks");
    
    // Sincronizador
    sync_ = std::make_shared<Synchronizer<ApproximateTimePolicy>>(
        ApproximateTimePolicy(10), image_sub_, landmarks_sub_);
    
    sync_->registerCallback(&AnimalFaceNode::callback, this);
    
    // Publisher
    image_pub_ = image_transport::create_publisher(this, "filtered_image", qos.get_rmw_qos_profile());
    
    // Hilo para entrada de teclado
    keyboard_thread_ = std::thread(&AnimalFaceNode::keyboard_listener, this);
    
    RCLCPP_INFO(get_logger(), "Nodo de filtros animales inicializado");
    RCLCPP_INFO(get_logger(), "Teclas: 1-Oso | 2-Gato | 3-Mono");
}

void AnimalFaceNode::callback(const ImageMsg::ConstSharedPtr& img_msg, 
                            const FaceLandmarks::ConstSharedPtr& landmarks_msg) {
    try {
        cv::Mat yuv_image(img_msg->height, img_msg->width, CV_8UC2, 
                        const_cast<uchar*>(img_msg->data.data()));
        cv::Mat frame;
        cv::cvtColor(yuv_image, frame, cv::COLOR_YUV2BGR_YUYV);
        
        std::vector<cv::Point2f> landmarks;
        for(const auto& point : landmarks_msg->landmarks) {
            landmarks.emplace_back(
                point.x * frame.cols,
                point.y * frame.rows
            );
        }
        
        // Aplicar filtro con mutex
        std::lock_guard<std::mutex> lock(filter_mutex_);
        cv::flip(frame, frame, 1);
        frame = current_filter_.apply_filter(frame, landmarks);
        
        auto output_msg = cv_bridge::CvImage(
            img_msg->header, "bgr8", frame).toImageMsg();
        image_pub_.publish(output_msg);
        
    } catch(const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error en callback: %s", e.what());
    }
}

void AnimalFaceNode::keyboard_listener() {
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while(running_) {
        int key = getchar();
        if(key_bindings_.find(key) != key_bindings_.end()) {
            std::lock_guard<std::mutex> lock(filter_mutex_);
            change_filter(key_bindings_[key]);
            RCLCPP_INFO(get_logger(), "Filtro cambiado a: %s", key_bindings_[key].c_str());
        }
    }
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

void AnimalFaceNode::change_filter(const std::string& animal) {
    try {
        current_filter_ = AnimalFilter(animal);
    } catch(const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error cambiando filtro: %s", e.what());
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AnimalFaceNode>();
    rclcpp::spin(node);
    node->shutdown();  // Llama al método público en lugar de acceder a miembros privados
    rclcpp::shutdown();
    return 0;
}