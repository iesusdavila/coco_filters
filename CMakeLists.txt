cmake_minimum_required(VERSION 3.8)
project(buddy-filters)

# ========== Configuración común ==========
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)

install(PROGRAMS
  scripts/face_landmark_detector.py
  scripts/FaceFilters.py
  scripts/BearFace.py
  scripts/CatFace.py
  scripts/MonkeyFace.py
  scripts/utils/AdvancedFaceFilter.py
  scripts/utils/FaceFilterBase.py
  scripts/utils/Filters.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Configuración para C++ ==========
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(buddy_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED) 
find_package(image_transport REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(face_filter_node 
  src/face_filter_node.cpp
  src/filters/GlassesFilter.cpp
  src/filters/FaceFilterBase.cpp
  # src/filters/HatFilter.cpp
  # src/filters/NoseFilter.cpp
  src/filters/MouthFilter.cpp
  # src/filters/FaceMaskFilter.cpp
)

target_include_directories(face_filter_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  include/
)

target_link_libraries(face_filter_node
  ${rclcpp_LIBRARIES}
  ${OpenCV_LIBS}
)

ament_target_dependencies(face_filter_node
  rclcpp
  sensor_msgs
  std_msgs
  OpenCV
  buddy_interfaces
  cv_bridge
  message_filters
  ament_index_cpp
  image_transport
)

install(TARGETS
  face_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Instalación común ==========
install(DIRECTORY
  scripts imgs src include
  DESTINATION share/${PROJECT_NAME}
)

# ========== Exportación de dependencias ==========
ament_export_dependencies(
  rclcpp
  rclpy
  sensor_msgs
  std_msgs
  control_msgs
  OpenCV
  buddy_interfaces
  cv_bridge
  message_filters
  ament_index_cpp
  image_transport
)

ament_export_include_directories(include)
ament_export_libraries(face_filter_node)

ament_package()