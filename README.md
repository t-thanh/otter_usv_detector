# otter_usv_detector

YOLO OBB detection and pose estimation of the Maritime Robotics Otter USV from a UAV gimbal camera, plus a Gazebo-based dataset acquisition and training pipeline.

Part of the [gazebo_uav_usv_landing](https://github.com/t-thanh/gazebo_uav_usv_landing) project — GPS-denied autonomous landing of an MRS X500 PX4 quadrotor on a moving Otter USV.

## Packages

| Package | Description |
|---|---|
| [`otter_usv_detector`](otter_usv_detector/) | YOLO OBB inference node, YOLO-based and ArUco/AprilTag pose estimators, offline evaluation pipeline |
| [`dataset_acquisition`](dataset_acquisition/) | Gazebo-based dataset collection (overhead + gimbal FK), annotation, verification, and YOLO training |

## Detection and pose estimation pipeline

```
Gremsy G-Hadron gimbal camera
        │  /uav1/overhead_cam/image_raw
        ▼
  detector_node.py          ← YOLO OBB inference (Ultralytics, conda: yolo_training)
        │  ObbDetectionArray → /usv_detection/result
        │
        ├──→  yolo_pose_estimator_node.py  ── long-range approach (YOLO OBB corners → USV pose)
        │
        └──→  aruco_pose_estimator_node.py  ── close-range landing (nested ArUco solvePnP)
                 (from ar_code_landing package)
```

**Handover strategy:** YOLO OBB guides the UAV into ArUco detection range. Once the nested fiducial board (outer AprilTag 36h11 ID=10, inner ArUco 4×4 ID=1) is visible, the ArUco estimator takes over for centimetre-accurate landing.

## Dataset acquisition pipeline

```
Gazebo simulation
  └─ collect_dataset_gimbal.py      ← teleports virtual camera via gimbal FK
        ├─ 1500 positives           ← random drone pos, gimbal yaw+pitch, USV visible
        ├─  150 hard negatives A    ← deck-level, gimbal horizontal/upward
        └─  350 hard negatives B    ← low altitude, gimbal sideways/upward
              │
              ▼  YOLO OBB label format (8 normalised corners, class 0 = otter_usv)
  yolo obb train data=otter_usv.yaml model=yolo11s-obb.pt imgsz=960
```

Corner order: front-port → front-starboard → rear-starboard → rear-port (TL→TR edge encodes bow direction).

## Repository structure

```
otter_usv_detector/        ← ROS package: detector + pose estimators + evaluation
  msg/                     ← ObbDetection.msg, ObbDetectionArray.msg
  scripts/                 ← detector_node.py, yolo_pose_estimator_node.py,
  │                           aruco_pose_test_node.py, eval_driver_node.py, …
  launch/                  ← pose_validation, detector, eval_pos_estimator, …
  config/                  ← detector.yaml, test_grid.yaml (24-scenario eval grid)
  test_samples/            ← ground-truth evaluation images (24 scenarios)

dataset_acquisition/       ← ROS package: Gazebo-based dataset collection + training
  scripts/                 ← collect_dataset_gimbal.py, annotator.py, verify_dataset.py
  launch/                  ← data_collection_gimbal.launch, data_collection.launch
  config/                  ← collection_params_gimbal.yaml, lighting_params.yaml
  urdf/                    ← virtual overhead_camera.urdf.xacro
  worlds/                  ← usv_uav_collect.world, open_water.world
  dataset_gimbal/          ← generated (gitignored) — images + YOLO labels
  runs/                    ← generated (gitignored) — YOLO training runs + weights
```

## Requirements

| Type | Requirement |
|---|---|
| OS / ROS | Ubuntu 20.04, ROS Noetic |
| Simulation | Gazebo 11, `usv_simulator`, `uav_gimbal` (from `gazebo_uav_usv_landing`) |
| Python (ROS) | `numpy`, `opencv-python`, `scipy` |
| Python (training) | conda env `yolo_training` — `ultralytics`, `torch` (Python ≥ 3.10) |

## Setup

```bash
cd ~/catkin_ws/src
git clone https://github.com/t-thanh/otter_usv_detector.git

# rosdep
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src --rosdistro=noetic -y

catkin build otter_usv_detector dataset_acquisition
source devel/setup.bash
```

The `detector_node.py` must run inside the `yolo_training` conda environment.
Use the provided wrapper:

```bash
conda activate yolo_training
pip install ultralytics rospkg catkin-pkg
```

`run_detector.sh` injects the ROS Python path into that environment so `rospy` is available.

## Model weights

Trained weights are **not** stored in this repository. Download from the
[Releases page](https://github.com/t-thanh/otter_usv_detector/releases) and update
`model_path` in `otter_usv_detector/config/detector.yaml`.

To retrain from scratch, see the [dataset_acquisition README](dataset_acquisition/README.md).

## Quick start

### Collect dataset and train

```bash
# 1. Collect gimbal-view dataset (Gazebo, ~30 min for 2000 images)
roslaunch dataset_acquisition data_collection_gimbal.launch gui:=false

# 2. Verify annotations
python3 src/dataset_acquisition/scripts/verify_dataset.py --n 50 --split all

# 3. Train (inside yolo_training conda env)
conda activate yolo_training
yolo obb train \
  data=<workspace>/src/dataset_acquisition/dataset_gimbal/otter_usv.yaml \
  model=yolo11s-obb.pt imgsz=960 epochs=100 name=Otter_YOLO_Gimbal
```

### Validate pose estimator

```bash
# Collect 24-scenario ground-truth test set (Gazebo)
roslaunch otter_usv_detector collect_test_samples.launch gui:=false

# Run offline accuracy evaluation (no Gazebo needed)
roslaunch otter_usv_detector eval_pos_estimator.launch
# → prints metrics table + saves test_samples/eval_results.yaml
```

### Live validation scenario

```bash
# Start the world (from gazebo_uav_usv_landing)
roslaunch otter_gazebo usv_uav_gimbal.launch

# Spawn the UAV, arm, take off to ~10 m, then:
roslaunch otter_usv_detector pose_validation.launch
rqt_image_view /yolo_pose/image
rostopic echo /yolo_pose/error
```

## Evaluation results

From the 24-scenario offline evaluation (gimbal-view dataset, `yolo11s-obb`, imgsz=960):

| Metric | Mean | Std | Max | Median |
|---|---|---|---|---|
| Lateral error [m] | 0.024 | 0.016 | 0.074 | 0.019 |
| 3D position error [m] | 0.025 | 0.016 | 0.076 | 0.020 |
| USV yaw error [°] | 2.1 | 1.8 | 6.3 | 1.7 |

22/24 samples: full 6-DOF pose. 2/24 samples: centroid-only fallback (low altitude, wide gimbal angle).
