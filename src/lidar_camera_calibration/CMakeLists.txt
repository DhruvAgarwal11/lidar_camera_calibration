cmake_minimum_required(VERSION 3.5)
project(lidar_camera_calibration)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)

# Uncomment this if the package has a setup.py. This macro ensures
# modules and global scripts declared therein get installed
# ament_python_install_package()

# Add include directories
include_directories(
  ${ament_INCLUDE_DIRS}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  PROGRAMS
    scripts/calibrate_camera_lidar.py
  DESTINATION lib/${PROJECT_NAME}
)


# Declare a C++ executable
#add_executable(${PROJECT_NAME}_node src/lidar_camera_calibration_node.cpp)

# Specify libraries to link a library or executable target against
#target_link_libraries(${PROJECT_NAME}_node
#  ${ament_LIBRARIES}
#)

#ament_target_dependencies(${PROJECT_NAME}_node
#  rclcpp
#  sensor_msgs
#  std_msgs
#  image_geometry
#  cv_bridge
#)

#install(TARGETS ${PROJECT_NAME}_node
#  DESTINATION lib/${PROJECT_NAME}
#)

# Mark executables and/or libraries for installation
#install(TARGETS
#  ${PROJECT_NAME}_node
#  ARCHIVE DESTINATION lib
#  LIBRARY DESTINATION lib
#  RUNTIME DESTINATION bin
#)

# Mark cpp header files for installation
#install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION include
#  FILES_MATCHING PATTERN "*.h"
#)

ament_package()

