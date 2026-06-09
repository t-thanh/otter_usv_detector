#!/bin/bash
# run_detector.sh
# ───────────────
# Wrapper used by detector.launch / test_image.launch as launch-prefix.
# Injects ROS Python packages into PYTHONPATH so that the yolo_training
# conda interpreter (Python 3.10) can import rospy, sensor_msgs, etc.
# cv_bridge is intentionally NOT added — the node uses numpy conversion instead.

WS=/home/t-thanh/Garage/uav_usv_sim
CONDA_PYTHON=/home/t-thanh/anaconda3/envs/yolo_training/bin/python3

export PYTHONPATH="\
/opt/ros/noetic/lib/python3/dist-packages:\
${WS}/devel/lib/python3/dist-packages:\
${WS}/devel/.private/otter_usv_detector/lib/python3/dist-packages:\
${PYTHONPATH}"

exec "${CONDA_PYTHON}" "$@"
