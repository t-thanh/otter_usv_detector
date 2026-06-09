# otter_usv_detector

YOLO OBB detector and pose estimators for the Otter USV.  Runs Ultralytics YOLO OBB inference on the Gremsy G-Hadron gimbal camera and publishes oriented bounding-box detections.  Two independent pose estimators are provided for comparison:

| Estimator | Input | Method | Primary use |
|-----------|-------|--------|-------------|
| **YOLO OBB** | Detected 2-D corners + gimbal FK + UAV odometry | Geometric back-projection to known AR-panel plane | Long-range approach navigation |
| **ArUco / AprilTag** | Nested fiducial markers + gimbal FK + UAV odometry | `solvePnP (IPPE_SQUARE)` per marker | Close-range precise landing |

**Handover strategy:** YOLO OBB guides the UAV to within ArUco detection range.  Once the nested fiducial board is visible, ArUco takes over for centimetre-accurate landing.

The YOLO estimator outputs the USV centre position and heading expressed in the camera optical frame (`/yolo_pose/usv_in_cam`).  It includes a centroid-only fallback mode at low altitudes where reproj-error gating would otherwise reject the pose.

---

## Overview

```
/uav1/overhead_cam/image_raw
        │
        ▼
  detector_node.py  (conda: yolo_training)
        │  ObbDetectionArray → /usv_detection/result
        │
        ├──→  yolo_pose_estimator_node.py  ─→  /yolo_pose/usv_in_cam   (PoseStamped)
        │       + /uav1/ground_truth                /yolo_pose/gt_usv_in_cam
        │       + /uav1/gimbal/joint_states          /yolo_pose/error   (XY/Z/3D [m])
        │       + /gazebo/model_states               /yolo_pose/image
        │
        └──→  (gimbal_usv_tracker package — IBVS tracking)

/uav1/overhead_cam/image_raw
        │
        ▼
  aruco_pose_test_node.py
        │  /aruco_test/outer/pose  (AprilTag 36h11)
        │  /aruco_test/inner/pose  (ArUco 4x4)
        └─ /aruco_test/error       (XY/Z/3D vs GT [m])
```

The detector runs inside the `yolo_training` conda environment (Ultralytics requires Python ≥ 3.10).  `run_detector.sh` injects ROS packages into that environment so the node can still use `rospy`.

---

## Dependencies

| Type | Requirement |
|------|-------------|
| ROS | `rospy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `gazebo_msgs`, `cv_bridge` |
| Python (system ROS) | `numpy`, `opencv-python`, `scipy` |
| Python (conda `yolo_training`) | `ultralytics`, `torch` |

---

## Building

```bash
catkin build otter_usv_detector
source devel/setup.bash
```

The build generates the `ObbDetection` and `ObbDetectionArray` message types used by all downstream packages (`gimbal_usv_tracker`, `yolo_pose_estimator_node`).

---

## Custom messages

### `ObbDetection`

| Field | Type | Description |
|-------|------|-------------|
| `class_name` | `string` | Detected class label |
| `confidence` | `float32` | Detection confidence `[0, 1]` |
| `cx`, `cy` | `float32` | Bounding box centre (pixels) |
| `width`, `height` | `float32` | OBB dimensions (pixels) |
| `angle_deg` | `float32` | OBB rotation (Ultralytics convention) |
| `corners` | `float32[8]` | Corner pixels `[x1,y1, x2,y2, x3,y3, x4,y4]` |

Corner order matches training labels: **front-port → front-starboard → rear-starboard → rear-port** (TL→TR edge encodes the bow direction).

### `ObbDetectionArray`

| Field | Type | Description |
|-------|------|-------------|
| `header` | `std_msgs/Header` | Timestamp + frame from source image |
| `image_width`, `image_height` | `uint32` | Source image dimensions |
| `detections` | `ObbDetection[]` | All detections in this frame (may be empty) |

---

## Nodes

### `detector_node.py` — YOLO OBB inference

Runs Ultralytics YOLO OBB inference on incoming camera frames and publishes detections.
Must be launched with the `run_detector.sh` prefix so it executes inside the `yolo_training` conda environment.

**Subscribed topics**

| Topic | Type | Description |
|-------|------|-------------|
| `<image_topic>` | `sensor_msgs/Image` | Raw camera frames |

**Published topics**

| Topic | Type | Description |
|-------|------|-------------|
| `<detection_topic>` | `ObbDetectionArray` | OBB detections per frame |
| `<viz_topic>` | `sensor_msgs/Image` | Annotated frame (optional) |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~model_path` | *(required)* | Absolute path to `best.pt` weights |
| `~image_topic` | `/overhead_cam/image_raw` | Input camera topic |
| `~detection_topic` | `/usv_detection/result` | Output detection topic |
| `~viz_topic` | `/usv_detection/image` | Output visualisation topic |
| `~conf_threshold` | `0.35` | Minimum confidence to publish |
| `~iou_threshold` | `0.45` | NMS IoU threshold |
| `~imgsz` | `960` | Inference image size (must match training) |
| `~device` | `"0"` | GPU index or `"cpu"` |
| `~publish_viz` | `true` | Publish annotated image |
| `~viz_line_width` | `2` | OBB polygon line thickness (px) |

---

### `yolo_pose_estimator_node.py` — Guidance-grade USV pose from YOLO OBB

Estimates the Otter USV centre position and heading in the camera optical frame for UAV approach navigation.  Once the ArUco board is visible, precise landing is handled by `aruco_pose_test_node`.

Uses geometric back-projection instead of `solvePnP` — the flat AR panel viewed from near-nadir is a degenerate case for PnP, but back-projection to a known-height plane is robust at all angles.

**Algorithm**

The gimbal FK chain gives the camera pose in world (`p_opt`, `R_opt`) exactly from drone odometry — no estimation needed.

1. **Centroid back-projection** — back-project the OBB centre pixel to the AR-panel plane (z = 0.70 m).  Always succeeds; used for the centroid-fallback path.

2. **USV yaw** — back-project the OBB `angle_deg` as a world-space axis direction (1-pixel step from OBB centre onto panel plane).  Try 4 candidates (base_yaw + k×90°).  For each, project all 4 object corners and use the Hungarian algorithm (min-assignment) to match projected vs. detected corners.  Select the candidate with lowest mean reproj error.

3. **USV base_link position** = panel_centre_world − R_usv × (−0.15, 0, 0.70).  Z correction (−0.70 m) is yaw-independent; result is at the waterline.

4. **Output in camera frame** — `usv_in_cam = R_opt⁻¹ × (usv_pos − p_opt)`.  Orientation carries the estimated USV yaw expressed in the camera frame.

5. **Centroid fallback** — if reproj error exceeds `max_reproj_err_px`, publish the centroid-only position (orientation = identity = yaw unknown).  This ensures the navigator always receives a position fix, even at low altitude where corner matching is noisy.

**AR-panel geometry** (Otter `base_link` frame, in metres — matches training annotations)

Source: `otter_base.urdf.xacro` `scale="2.5 2.5 0.05"` and `collection_params_gimbal.yaml` `half_size_m=1.25`.

```
center: (−0.15, 0.0, 0.70)   half_size: 1.25  (2.5×2.5 m panel)
corners:  fp(+1.10, +1.25)   fs(+1.10, −1.25)
          rs(−1.40, −1.25)   rp(−1.40, +1.25)
```

**Subscribed topics**

| Topic | Type | Description |
|-------|------|-------------|
| `<detection_topic>` | `ObbDetectionArray` | YOLO OBB detections |
| `/<ns>/overhead_cam/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics (latched) |
| `/<ns>/overhead_cam/image_raw` | `sensor_msgs/Image` | Camera frames (for debug viz) |
| `/<ns>/ground_truth` | `nav_msgs/Odometry` | UAV world pose |
| `/<ns>/gimbal/joint_states` | `sensor_msgs/JointState` | Gimbal yaw/roll/pitch angles |
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | USV ground-truth pose |

**Published topics**

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo_pose/usv_in_cam` | `geometry_msgs/PoseStamped` | USV centre in camera optical frame (position + heading) |
| `/yolo_pose/gt_usv_in_cam` | `geometry_msgs/PoseStamped` | Ground-truth USV in camera frame |
| `/yolo_pose/error` | `geometry_msgs/PointStamped` | `.x`=lateral err, `.y`=range err, `.z`=3D err [m] |
| `/yolo_pose/image` | `sensor_msgs/Image` | Debug: detected + reprojected corners |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~gimbal_ns` | `uav1` | UAV namespace — resolves all `/<ns>/…` topics |
| `~usv_model_name` | `otter` | Gazebo model name for ground-truth lookup |
| `~detection_topic` | `/usv_detection/result` | Input detection topic |
| `~base_x/y/z_offset` | `0.10/0/0` | Gimbal base offset from FCU in body frame [m] |
| `~conf_threshold` | `0.50` | Minimum detection confidence to attempt PnP |
| `~max_reproj_err_px` | `25.0` | Reject solution if mean reprojection error exceeds this [px] |
| `~altitude_tol_m` | `5.0` | Cyclic-assignment gate: max `|cam_z − drone_z|` [m] |
| `~hfov_deg` | `67.0` | Horizontal FOV fallback (overridden by `camera_info`) |
| `~image_width` / `~image_height` | `924` / `690` | Image size fallback |

---

### `aruco_pose_test_node.py` — Nested fiducial marker pose test

Detects the nested fiducial board (Target_2_board.png) on the Otter USV using the Gremsy G-Hadron gimbal camera.  Runs two detectors per frame:

- **Outer** — `DICT_APRILTAG_36h11` ID=10 (2.000 m physical side, green overlay)
- **Inner** — `DICT_4X4_50` ID=1 (0.200 m physical side, orange overlay)

For each detected marker, pose is estimated via `solvePnP (IPPE_SQUARE)` and transformed to world frame using the full gimbal FK chain.  The result is compared against Gazebo `model_states` ground truth and reported as XY, Z, and 3-D Euclidean error.

**Subscribed topics**

| Topic | Type | Description |
|-------|------|-------------|
| `/<ns>/overhead_cam/image_raw` | `sensor_msgs/Image` | Gimbal camera frames |
| `/<ns>/overhead_cam/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics (latched) |
| `/<ns>/ground_truth` | `nav_msgs/Odometry` | Drone world pose |
| `/<ns>/gimbal/joint_states` | `sensor_msgs/JointState` | Gimbal yaw/roll/pitch angles |
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | USV ground-truth pose |

**Published topics**

| Topic | Type | Description |
|-------|------|-------------|
| `/aruco_test/image` | `sensor_msgs/Image` | Annotated frame (green=AprilTag, orange=ArUco) |
| `/aruco_test/outer/pose` | `geometry_msgs/PoseStamped` | Outer AprilTag world pose |
| `/aruco_test/inner/pose` | `geometry_msgs/PoseStamped` | Inner ArUco world pose |
| `/aruco_test/error` | `geometry_msgs/PointStamped` | `.x`=XY err, `.y`=Z err, `.z`=3D err [m] |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~gimbal_ns` | `uav1` | UAV/gimbal namespace |
| `~usv_model_name` | `otter` | Gazebo model name for ground truth |
| `~marker_size_m` | `0.2` | Inner ArUco physical side [m] |
| `~outer_marker_size_m` | `2.0` | Outer AprilTag physical side [m] |
| `~base_x/y/z_offset` | `0.10/0/0` | Gimbal base offset from FCU [m] |

---

### `pos_estimator_node.py` — USV 3-D position from nadir OBB

Back-projects the OBB centroid through a pinhole camera model onto the water surface plane to recover the USV world position.  Assumes a **fixed nadir** camera mount — for the movable gimbal camera use `yolo_pose_estimator_node` or `gimbal_usv_tracker` instead.

**Subscribed topics**

| Topic | Type | Description |
|-------|------|-------------|
| `~detection_topic` | `ObbDetectionArray` | Detections from `detector_node` |
| `~odom_topic` | `nav_msgs/Odometry` | UAV world pose |

**Published topics**

| Topic | Type | Description |
|-------|------|-------------|
| `~output_topic` | `geometry_msgs/PoseStamped` | USV position in world ENU |
| `~debug_topic` | `geometry_msgs/PointStamped` | Altitude cross-check values |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~camera_hfov_deg` | `67.0` | Horizontal FOV [deg] |
| `~image_width` / `~image_height` | `924` / `690` | Image dimensions [px] |
| `~cam_body_x/y/z` | `0.10/0/−0.046` | Camera offset from FCU [m] |
| `~usv_length` | `2.0` | Otter USV length for altitude cross-check [m] |
| `~water_z` | `0.0` | Water surface altitude [m] |
| `~min_confidence` | `0.30` | Detection confidence gate |

---

### `gimbal_nadir_cmd.py` — One-shot gimbal nadir command

Waits a configurable delay, then sends `pitch=+π/2` (nadir) and `yaw=0` to the gimbal position controller.  Used by `pose_validation.launch` so the camera is pointing at the USV before the operator arms and climbs.

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~ns` | `uav1` | UAV namespace |
| `~delay_s` | `12.0` | Seconds to wait after node start |

---

### `image_publisher_node.py` — Offline image streamer

Publishes saved images (single file or directory) as `sensor_msgs/Image` for offline testing.

---

### `collect_test_samples.py` — Ground-truth test set collector

Uses the gimbal FK chain to compute each scenario's optical camera pose, teleports the `overhead_cam` URDF model to that pose in Gazebo, captures a rendered frame, and saves the image together with the full 6-DOF ground truth (camera position + drone orientation in USV frame).

Run via `collect_test_samples.launch` — not directly.

---

### `eval_driver_node.py` — Offline USV-in-cam evaluation driver

Replays saved test-sample images one by one, publishing synthetic drone state (Odometry, JointState, ModelStates, CameraInfo) so `yolo_pose_estimator_node` behaves as if a live UAV were flying.

Ground-truth USV position in camera frame is computed per sample via the same gimbal FK, then compared against the `/yolo_pose/usv_in_cam` estimate.  Centroid-fallback samples are reported separately in the summary (yaw error = N/A).

**Reported metrics** (per sample + aggregate mean / std / max / median)

| Metric | Unit | Description |
|--------|------|-------------|
| `err_x`, `err_y` | m | Lateral position error (image-plane X, Y in camera frame) |
| `err_z` | m | Range / depth error |
| `err_lat` | m | Lateral error magnitude √(err_x²+err_y²) |
| `err_3d` | m | Full 3-D Euclidean position error |
| `err_dist` | m | Error in camera-to-USV range magnitude |
| `err_yaw` | ° | USV yaw error in world frame (NaN for centroid-fallback poses) |

Results are written to `<test_samples_dir>/eval_results.yaml` on completion.

---

### `verify_nested_tags.py` — Offline board detection checker

Standalone script (no ROS) that verifies both markers in a nested board image are detectable by OpenCV.  Saves annotated output images.

```bash
python3 scripts/verify_nested_tags.py /path/to/board.png
python3 scripts/verify_nested_tags.py   # synthesises a reference board
```

---

## Launch files

### `pose_validation.launch` — Side-by-side estimator comparison (live)

Full validation scenario: USV at origin, UAV spawned on a platform 5 m away, both pose estimators and Gazebo ground truth running simultaneously.

```bash
roslaunch otter_usv_detector pose_validation.launch [gui:=true]
```

**Scenario**

- Otter USV at world origin.
- 2×2×1 m static platform at (5, 0, 0) — top face at z=1 m.
- X500 UAV spawned at `--pos 5 0 2 0` (on top of platform).
- `gimbal_nadir_cmd` sends `pitch=+π/2` (nadir) after 12 s warm-up.
- After the nadir command fires, arm and climb to ≈8 m — USV enters the camera FOV.

**Post-launch procedure**

```bash
roslaunch otter_gazebo start_uav.launch
# or via MRS aliases in the uav1 tmux session:
#   mrs arm  →  mrs takeoff
# Then climb to 8–10 m:
rosservice call /uav1/control_manager/goto "goal: [0, 0, 10.0, 0.0]"
```

**Key output topics**

| Topic | Description |
|-------|-------------|
| `/yolo_pose/error` | YOLO OBB pose error `.x`=XY `.y`=Z `.z`=3D [m] |
| `/aruco_test/error` | ArUco pose error (same format) |
| `/yolo_pose/usv_in_cam` | USV centre in camera frame (position + heading) |
| `/yolo_pose/gt_usv_in_cam` | Ground-truth USV in camera frame |
| `/yolo_pose/image` | Debug: detected + reprojected OBB corners |
| `/aruco_test/image` | Debug: ArUco / AprilTag detections |

```bash
rostopic echo /yolo_pose/error
rostopic echo /aruco_test/error
rqt_image_view /yolo_pose/image
rqt_image_view /aruco_test/image
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `gui` | `true` | Show Gazebo GUI |
| `verbose` | `false` | Verbose Gazebo output |

---

### `collect_test_samples.launch` — Ground-truth test set collection

Spawns Gazebo (`usv_uav_collect.world`) with the Otter USV and the `overhead_cam` URDF, then runs `collect_test_samples.py` through all 24 deterministic scenarios in `test_grid.yaml`.

```bash
roslaunch otter_usv_detector collect_test_samples.launch [gui:=false]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `gui` | `true` | Show Gazebo GUI |
| `verbose` | `false` | Verbose Gazebo output |
| `output_dir` | *(pkg/test_samples)* | Where to save images + metadata |
| `test_grid` | *(pkg/config/test_grid.yaml)* | Path to scenario grid YAML |

Output: `<output_dir>/images/sample_NNNN.jpg` and `<output_dir>/metadata.yaml`.

---

### `eval_pos_estimator.launch` — Offline full-pose evaluation

No Gazebo needed.  Replays the collected samples through the full estimator pipeline and prints + saves accuracy metrics.

```bash
roslaunch otter_usv_detector eval_pos_estimator.launch [test_samples_dir:=] [conf:=0.35]
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `test_samples_dir` | *(pkg/test_samples)* | Directory produced by collect step |
| `ns` | `uav1` | UAV namespace (must match collect step) |
| `device` | `0` | GPU index or `"cpu"` |
| `conf` | `0.35` | YOLO confidence threshold |
| `detect_timeout_s` | `8.0` | Max seconds to wait per sample |

Results saved to `<test_samples_dir>/eval_results.yaml`.

---

### `aruco_pose_test.launch` — ArUco estimator only

```bash
roslaunch otter_usv_detector aruco_pose_test.launch [ns:=uav1] [show_image:=true]
```

Prerequisite: `usv_uav_gimbal.launch` and `gimbal_controllers.launch` already running; UAV airborne with gimbal pointing at the USV.

---

### `detector.launch` — Detection only

```bash
roslaunch otter_usv_detector detector.launch \
  [image_topic:=/uav1/overhead_cam/image_raw] \
  [device:=0] [conf:=0.35] [publish_viz:=true]
```

Default `image_topic` is `/overhead_cam/image_raw` (standalone / overhead camera).  Override for the gimbal UAV scenario as shown above.

---

### `test_image.launch` — Offline test on saved images

```bash
# Single image:
roslaunch otter_usv_detector test_image.launch \
    image_path:=/path/to/image.jpg

# Whole gimbal test split (loops):
roslaunch otter_usv_detector test_image.launch \
    image_path:=/home/t-thanh/Garage/uav_usv_sim/src/dataset_acquisition/dataset_gimbal/images/test \
    [rate_hz:=1.0] [loop:=true]
```

Starts an image publisher alongside the detector — no Gazebo needed.

---

## Testing the pose estimator

### Step 1 — Collect the test samples (runs Gazebo, ~5 min)

```bash
source devel/setup.bash
roslaunch otter_usv_detector collect_test_samples.launch gui:=false
```

The node teleports the overhead camera and USV through all 24 scenarios, captures a rendered image for each, checks that all AR-panel corners are in-frame, and writes:

```
src/otter_usv_detector/test_samples/
  images/
    sample_0000.jpg  …  sample_0023.jpg
  metadata.yaml          ← full 6-DOF ground truth per sample
```

Scenarios that fail the visibility check are skipped and logged as warnings.  The launch exits automatically when collection is complete.

### Step 2 — Run the offline evaluation (no Gazebo)

```bash
roslaunch otter_usv_detector eval_pos_estimator.launch
```

The pipeline is:

```
eval_driver_node  →  detector_node  →  yolo_pose_estimator_node
     (images +            (YOLO OBB)       (geometric back-projection)
     synthetic state)
          └────────────── /yolo_pose/usv_in_cam ◄──────────────┘
                          compared vs. metadata.yaml GT
```

For each sample the driver:
1. Publishes synthetic Odometry, JointState, ModelStates, CameraInfo.
2. Publishes the saved image on `/uav1/overhead_cam/image_raw`.
3. Waits up to `detect_timeout_s` for `/yolo_pose/usv_in_cam`.
4. Computes and logs: lateral/range/3D position error + USV yaw error.

### Step 3 — Read the results

**Console output** (per sample):

```
[eval_driver|FULL]     #02 "alt_20"  pos(-0.012,+0.003,-0.002)m lat=0.012m 3D=0.012m dist_err=0.002m  yaw_err=-1.1°  conf=0.97
[eval_driver|CENTROID] #00 "alt_05"  pos(-0.021,+0.008,+0.013)m lat=0.022m 3D=0.026m dist_err=0.008m  yaw=N/A  conf=0.70
```

**Aggregate summary** (printed on completion):

```
══════════════════════════════════════════════════════════════════
 YOLO OBB POSE EVALUATION  24/24 pose OK  (22 full + 2 centroid)  24/24 detected
──────────────────────────────────────────────────────────────────
  Metric        mean     std      max      median
  x_cam [m]    0.018   0.012    0.062     0.014
  y_cam [m]    0.014   0.010    0.051     0.011
  z_cam [m]    0.006   0.005    0.021     0.004
  lat [m]      0.024   0.016    0.074     0.019
  3D [m]       0.025   0.016    0.076     0.020
  dist [m]     0.006   0.005    0.021     0.004
  yaw [°]      2.1     1.8      6.3       1.7     (22 full-pose samples)
══════════════════════════════════════════════════════════════════
```

**YAML output** — `test_samples/eval_results.yaml`:

```yaml
n_total: 24
n_detected: 24
n_pose_ok: 24
n_full_pose: 22
n_centroid_fallback: 2
summary:
  err_lat: {mean: 0.024, std: 0.016, max: 0.074, median: 0.019}
  err_3d:  {mean: 0.025, std: 0.016, max: 0.076, median: 0.020}
  err_yaw: {mean: 2.1,   std: 1.8,   max: 6.3,   median: 1.7  }
  …
results:
  - sample_id: 0
    label: alt_05
    centroid_only: true
    gt_x: 0.00  gt_y: 0.03  gt_z: 5.00
    est_x: -0.02  est_y: 0.04  est_z: 5.01
    err_lat: 0.022  err_3d: 0.026  err_yaw: .nan
  …
```

### Optional: custom output directory

```bash
roslaunch otter_usv_detector collect_test_samples.launch \
    output_dir:=/tmp/my_eval  gui:=false

roslaunch otter_usv_detector eval_pos_estimator.launch \
    test_samples_dir:=/tmp/my_eval
```

### Optional: monitor detection images live

```bash
rqt_image_view /usv_detection/image    # annotated YOLO OBB boxes
rqt_image_view /yolo_pose/image        # detected + reprojected corners
```

---

## Configuration

### `config/detector.yaml`

```yaml
model_path: "/home/t-thanh/Garage/uav_usv_sim/src/dataset_acquisition/runs/obb/Nested_Otter_YOLO2/weights/best.pt"

image_topic:      "/overhead_cam/image_raw"
detection_topic:  "/usv_detection/result"
viz_topic:        "/usv_detection/image"

conf_threshold:  0.35
iou_threshold:   0.45
imgsz:           960
device:          "0"
publish_viz:     true
```

Update `model_path` after retraining.  All gimbal-scenario launch files override `image_topic` to `/<ns>/overhead_cam/image_raw`.

### `config/test_grid.yaml`

24 deterministic scenarios across 5 groups:

| Group | Scenarios | Variable |
|-------|-----------|----------|
| `alt_*` | 6 | Altitude sweep (5–70 m), drone directly above USV |
| `rng_*` | 5 | Lateral range sweep (5–25 m), fixed alt=20 m |
| `dir_*` | 4 | Cardinal directions (±X, ±Y) at same geometry |
| `yaw_*` | 4 | Drone yaw 0/90/180/270°, same position |
| `mix_*` | 5 | Oblique, high-offset, compound gimbal angles |

`gimbal_pitch_deg: 90` = nadir (straight down).  `gimbal_yaw_deg` is in drone body frame.

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/detector_node.py` | YOLO OBB ROS inference node (conda-wrapped) |
| `scripts/yolo_pose_estimator_node.py` | 6-DOF pose estimator from YOLO OBB corners via solvePnP |
| `scripts/aruco_pose_test_node.py` | Nested ArUco/AprilTag detection & pose error analysis |
| `scripts/gimbal_nadir_cmd.py` | One-shot delayed nadir gimbal command |
| `scripts/pos_estimator_node.py` | Nadir-only USV position estimator (legacy) |
| `scripts/image_publisher_node.py` | Publishes saved images for offline testing |
| `scripts/collect_test_samples.py` | Gimbal FK-based 24-scenario ground-truth collector |
| `scripts/eval_driver_node.py` | Full 6-DOF offline pose estimator evaluation driver |
| `scripts/verify_nested_tags.py` | Offline script: verifies both markers detectable in board PNG |
| `scripts/run_detector.sh` | Conda environment wrapper for detector node |
| `launch/pose_validation.launch` | Full validation: platform + UAV + both estimators + GT |
| `launch/aruco_pose_test.launch` | ArUco/AprilTag estimator standalone |
| `launch/detector.launch` | YOLO detector only |
| `launch/test_image.launch` | Offline detector test on saved images |
| `launch/collect_test_samples.launch` | Gazebo-based 24-scenario test sample collection |
| `launch/eval_pos_estimator.launch` | Offline 6-DOF accuracy evaluation (no Gazebo) |
| `models/validation_platform.sdf` | Static 2×2×1 m platform for validation scenario |
| `msg/ObbDetection.msg` | Single OBB detection message |
| `msg/ObbDetectionArray.msg` | Per-frame detection array message |
| `config/detector.yaml` | Detector + pose estimator parameters |
| `config/test_grid.yaml` | 24-scenario deterministic evaluation grid |

---

## Nested fiducial board (Target_2_board.png)

The Otter USV AR panel uses a composite 2500×2500 px PNG with:

| Marker | Dictionary | ID | Size in texture | Physical size |
|--------|------------|----|-----------------|---------------|
| Outer | `DICT_APRILTAG_36h11` | 10 | 2000×2000 px (+ 250 px white border) | 2.000 m |
| Inner | `DICT_4X4_50` | 1 | 200×200 px (+ 33 px white quiet zone) | 0.200 m |

Inner marker centre is at (1000, 500) px within the AprilTag area.  Both padding conditions are required for reliable detection — verified by `verify_nested_tags.py`.  URDF `rpy="0 0 M_PI/2"` orients the board so the USV bow (+X) faces the top of the panel when viewed from above.

---

## Gimbal FK chain

All nodes (`aruco_pose_test_node`, `yolo_pose_estimator_node`, `collect_test_samples`, `gimbal_position_node`) share the identical FK chain:

```
drone body
  └─ + base_offset (0.10, 0, 0)
       └─ Rz(yaw_joint)   + yaw_link_offset   (0, 0, −0.025)
            └─ Rx(roll_joint)  + roll_link_offset  (0, 0, −0.030)
                 └─ Ry(pitch_joint) + pitch_link_offset (0, 0, −0.025)
                      └─ + opt_offset (0.025, 0, 0)
                           └─ Rz(−π/2) · Rx(−π/2)  →  optical frame
```

Gimbal convention: `pitch_joint = +π/2` = nadir (camera pointing straight down).
