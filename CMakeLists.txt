cmake_minimum_required(VERSION 3.8)
project(buddy-filters)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)

install(TARGETS ${EXEC_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

# Instalar el script Python
install(PROGRAMS
  scripts/FaceFilters.py
  scripts/BearFace.py
  scripts/CatFace.py
  scripts/MonkeyFace.py
  scripts/utils/AdvancedFaceFilter.py
  scripts/utils/FaceFilterBase.py
  scripts/utils/Filters.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Instalación común ==========
install(DIRECTORY
  scripts imgs
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp
  rclpy
  sensor_msgs
  buddy_interfaces
  std_msgs
)

ament_package()