cmake_minimum_required(VERSION 3.0.2)
project(otter_usv_detector)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  gazebo_msgs
  cv_bridge
)

# ── Custom messages ──────────────────────────────────────────────────────────
add_message_files(
  FILES
  ObbDetection.msg
  ObbDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    gazebo_msgs
    cv_bridge
)

# ── Install Python scripts ───────────────────────────────────────────────────
catkin_install_python(
  PROGRAMS
    scripts/detector_node.py
    scripts/image_publisher_node.py
    scripts/pos_estimator_node.py
    scripts/collect_test_samples.py
    scripts/eval_driver_node.py
    scripts/aruco_pose_test_node.py
    scripts/yolo_pose_estimator_node.py
    scripts/gimbal_nadir_cmd.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/run_detector.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ── Install launch, config, and models ──────────────────────────────────────
install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
install(DIRECTORY config/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)
install(DIRECTORY models/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models)
