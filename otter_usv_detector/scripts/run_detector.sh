#!/bin/bash
# run_detector.sh
# ───────────────
# Wrapper used by detector.launch / pose_validation.launch / test_image.launch /
# eval_pos_estimator.launch as launch-prefix.  It runs detector_node.py under a
# chosen Python interpreter while keeping the ROS Python packages importable.
#
# Interpreter selection (in priority order):
#   1. $DETECTOR_PYTHON   — explicit override (e.g. a conda env interpreter)
#   2. python3            — system interpreter (default; used inside Docker)
#
# On a workstation that runs the detector from a conda env, export e.g.:
#   export DETECTOR_PYTHON=$HOME/anaconda3/envs/yolo_training/bin/python3
# Inside the reproducibility Docker image, ultralytics is installed into the
# system python3, so no override is needed.
#
# cv_bridge is intentionally NOT required — the node uses numpy conversion.

DETECTOR_PYTHON="${DETECTOR_PYTHON:-python3}"

# Make ROS + this workspace's generated Python packages importable regardless of
# which interpreter we exec (a non-default interpreter won't have them on its
# default path).  Paths are derived from the sourced ROS environment.
EXTRA_PP="/opt/ros/${ROS_DISTRO:-noetic}/lib/python3/dist-packages"
if [ -n "${CMAKE_PREFIX_PATH}" ]; then
  # Prepend each catkin result-space's python dist-packages.
  IFS=':' read -ra _prefixes <<< "${CMAKE_PREFIX_PATH}"
  for _p in "${_prefixes[@]}"; do
    EXTRA_PP="${_p}/lib/python3/dist-packages:${EXTRA_PP}"
  done
fi

export PYTHONPATH="${EXTRA_PP}:${PYTHONPATH}"

exec "${DETECTOR_PYTHON}" "$@"
