cmake_minimum_required(VERSION 3.0.2)
project(dataset_acquisition)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_msgs
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/collect_dataset.py
  scripts/collect_dataset_gimbal.py
  scripts/annotator.py
  scripts/verify_dataset.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config worlds urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
