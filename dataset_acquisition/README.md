# dataset_acquisition

Automated Gazebo-based dataset collection pipeline for training a YOLO OBB detector to identify the Otter USV from aerial UAV imagery.

Two collection pipelines are provided, sharing the same annotation core (`annotator.py`):

| Pipeline | Script | Launch file | Camera model |
|----------|--------|-------------|--------------|
| **Overhead** — standalone camera, fixed nadir to 45° tilt | `collect_dataset.py` | `data_collection.launch` | Simple overhead URDF |
| **Gimbal** — simulates Gremsy G-Hadron on MRS X500 | `collect_dataset_gimbal.py` | `data_collection_gimbal.launch` | Same URDF, FK-computed pose |

---

## Overview

### Overhead pipeline

```
Gazebo (open_water.world)
  └─ overhead_cam URDF  ──→  /overhead_cam/image_raw
       └─ collect_dataset.py
            ├─ randomise camera alt / tilt / yaw / XY
            ├─ back-project target pixel → USV world position
            ├─ teleport USV + camera (yaw·tilt quaternion)
            ├─ save image  → dataset/images/{train,val,test}/
            └─ annotate    → dataset/labels/{train,val,test}/
```

### Gimbal pipeline

```
Gazebo (usv_uav_collect.world)
  └─ overhead_cam URDF  ──→  /overhead_cam/image_raw
       └─ collect_dataset_gimbal.py
            ├─ sample drone pos + gimbal joint angles (yaw, pitch)
            ├─ compute camera optical pose via FK (gimbal_position_node chain)
            ├─ teleport virtual camera to FK pose (full quaternion)
            ├─ USV fixed at world origin — no teleport needed
            ├─ save image  → dataset_gimbal/images/{train,val,test}/
            └─ annotate    → dataset_gimbal/labels/{train,val,test}/
                             (empty .txt for hard-negative backgrounds)
```

No real UAV, gimbal, or MRS stack is launched.  The collector purely moves a virtual
camera model and reads back the rendered Gazebo frame.  The world matches the actual
test environment (`usv_uav.world`) — same ocean, sky, sun, and ground plane — without
the MRS UAV plugins that require a running UAV stack.

---

## Dependencies

| Type | Requirement |
|------|-------------|
| ROS | `rospy`, `gazebo_msgs`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `cv_bridge` |
| ROS packages | `otter_gazebo`, `usv_worlds` |
| Python | `numpy`, `scipy`, `opencv-python` |
| Training | Ultralytics YOLOv8/v11 (conda env `yolo_training`) |

---

## Building

```bash
catkin build dataset_acquisition
source devel/setup.bash
```

---

## Overhead pipeline

### Launch

```bash
roslaunch dataset_acquisition data_collection.launch
```

| Argument | Default | Description |
|----------|---------|-------------|
| `gui` | `true` | Show Gazebo GUI |
| `verbose` | `false` | Verbose Gazebo output |
| `preview_only` | `false` | Collect 14-sample structured preview grid only |
| `params_file` | `config/collection_params.yaml` | Collection parameters |

### Key parameters (`config/collection_params.yaml`)

```yaml
output:
  n_samples: 1500        # total images to collect
  train_ratio: 0.70
  val_ratio:   0.20

pose:
  altitude_min_m:        5.0   # camera altitude range [m]
  altitude_max_m:       70.0
  tilt_min_deg:          0.0   # tilt from nadir [deg]
  tilt_max_deg:         45.0
  camera_spawn_bounds_m: 30.0  # XY offset of camera from origin

ar_marker:
  half_size_m: 1.0             # AR marker half-width on USV roof [m]
```

The back-projection strategy ensures the AR marker centre lands at a uniformly
sampled target pixel, giving balanced spatial coverage across the image.

---

## Gimbal pipeline

### Sampling strategy

Three collection modes, set via the `mode` argument:

| Mode | Count | Description |
|------|-------|-------------|
| **Positive** | 1500 | UAV above USV, gimbal pitch ∈ [0°, 90°] (horizontal to nadir), USV visible in frame |
| **HN-A** | 150 | Deck-level (0.5–1.5 m), gimbal horizontal/upward, USV NOT in frame |
| **HN-B** | 350 | Low altitude (2–15 m), gimbal pointing sideways/up, USV NOT in frame |

**Positive sampling:**
1. Random drone position (±30 m XY, 5–70 m altitude) and yaw.
2. Compute nominal gimbal yaw + pitch to aim at the USV AR marker centre (analytical inverse FK).
3. Add jitter (±0.3 rad) to distribute the bounding box across the image plane.
4. Clamp pitch to [0, π/2] — no pointing above horizontal for positives.
5. Reject if USV is not visible (jitter pushed it outside FOV).

**HN-A sampling:**
1. Drone at (0, 0, 0.5–1.5 m) — right on or just above the deck.
2. Gimbal yaw drawn from nominal 60°-increment directions (0/60/120/180/240/300°) ± 10° jitter.
3. Gimbal pitch in [−60°, 0°] (upward to horizontal) — never looking at the deck.
4. Reject if USV accidentally appears in frame.

**HN-B sampling:**
1. Drone at (0, 0, 2–15 m) with random yaw.
2. Gimbal yaw fully random, pitch in [−90°, 45°].
3. Reject if USV appears in frame.

Hard-negative label files are written empty (YOLO interprets as background).

### Launch

```bash
# Full collection (positive + hard negatives)
roslaunch dataset_acquisition data_collection_gimbal.launch

# Preview only — verify all 8 structured samples, then exit
roslaunch dataset_acquisition data_collection_gimbal.launch preview_only:=true

# Positives only
roslaunch dataset_acquisition data_collection_gimbal.launch mode:=positive

# Hard negatives only
roslaunch dataset_acquisition data_collection_gimbal.launch mode:=hard_negative
```

| Argument | Default | Description |
|----------|---------|-------------|
| `gui` | `true` | Show Gazebo GUI |
| `preview_only` | `false` | Collect structured preview grid and exit |
| `mode` | `all` | `positive` / `hard_negative` / `all` |
| `params_file` | `config/collection_params_gimbal.yaml` | Collection parameters |

### Preview grid

The preview grid has 9 deterministic samples covering all failure modes:

| Label | Drone altitude | Gimbal yaw | Gimbal pitch | Expect |
|-------|---------------|-----------|--------------|--------|
| `pos_nadir_10m` | 10 m, above USV | 0° | 90° (nadir) | USV at centre |
| `pos_nadir_50m` | 50 m, above USV | 0° | 90° (nadir) | USV small but visible |
| `pos_tilt_20m` | 20 m, 10 m XY offset | 180° | 63° | USV off-centre |
| `pos_tilt_side` | 20 m, 10 m Y offset | 270° | 63° | USV off-centre |
| `pos_oblique` | 30 m, 20 m XY offset | 225° | 43° | USV near edge |
| `hn_a_fwd_horiz` | 1 m (deck) | 0° | 0° (horizontal) | No USV — ocean/sky |
| `hn_a_left_up` | 0.8 m (deck) | 90° | −30° (upward) | No USV — sky |
| `hn_a_bwd_horiz` | 1 m (deck) | 180° | 0° (horizontal) | No USV — ocean |
| `hn_b_5m_up` | 5 m | 45° | −20° (upward) | No USV — sky |

Preview images are saved to `dataset_gimbal/preview/` as `<label>_raw.jpg` and `<label>_obb.jpg`.
Positive samples have corner-indexed OBB overlays (1=front-port … 4=rear-port).
A warning is printed for any sample that does not meet its `expect_usv` flag.

### Key parameters (`config/collection_params_gimbal.yaml`)

```yaml
output:
  n_positive: 1500
  n_hn_a: 150        # deck-level hard negatives
  n_hn_b: 350        # off-target hard negatives

gimbal:
  base_offset_m: [0.10, 0.00, 0.00]   # matches gimbal_position_node.py

ar_marker:
  center_x_m: -0.15
  center_y_m:  0.0
  center_z_m:  0.7
  half_size_m: 1.25   # 2.5 m panel (scale="2.5 2.5 0.05")

positive:
  altitude_min_m:    5.0
  altitude_max_m:   70.0
  drone_xy_bound_m: 30.0
  pitch_jitter_rad:  0.3
  yaw_jitter_rad:    0.3
```

### FK implementation

The collector uses the identical gimbal kinematic chain as `gimbal_position_node.py`
and `aruco_pose_test_node.py`:

```
base_link  →  yaw_link  →  roll_link  →  camera_link  →  optical
     Rz(yaw)       Rx(roll)        Ry(pitch)      Rz(−π/2)·Rx(−π/2)
```

The virtual camera URDF (base_link → cam_link via Ry(π/2) → optical via R_OPT) is
teleported so that its optical frame aligns with the FK result:

```
R_base = R_opt · R_OPT⁻¹ · Ry(−π/2)
```

Annotation uses `project_corners_fk()` from `annotator.py`, which takes the full
`R_opt` rotation rather than the simplified `cam_yaw + cam_tilt` model.

---

## Output structure

Both pipelines write the same directory layout:

```
dataset[_gimbal]/
├── images/
│   ├── train/   (70 %)
│   ├── val/     (20 %)
│   └── test/    (10 %)
├── labels/      (mirrored structure; empty .txt = background)
└── otter_usv.yaml
```

---

## Dataset verification

```bash
python3 scripts/verify_dataset.py --n 50 --cols 10 --split all
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n` | `50` | Number of images to display |
| `--cols` | `10` | Grid columns |
| `--split` | `all` | `train` / `val` / `test` / `all` |
| `--seed` | random | Fixed random seed for reproducibility |
| `--save` | — | Save the preview grid to a file |

---

## Annotation format

YOLO OBB corner format — 8 normalised coordinates, one object per line:

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

All coordinates are normalised to `[0, 1]`.  Class `0` = `otter_usv`.
Corner order: front-port → front-starboard → rear-starboard → rear-port
(TL→TR edge encodes the bow direction).
Empty label file = background image (no USV), used for hard negatives.

---

## Training

After collection, train with Ultralytics inside the `yolo_training` conda environment:

```bash
conda activate yolo_training
yolo obb train \
  data=<path_to_dataset_gimbal/otter_usv.yaml> \
  model=yolo11s-obb.pt \
  imgsz=960 \
  epochs=100 \
  name=Otter_YOLO_Gimbal
```

Trained weights land in:
```
dataset_gimbal/runs/obb/Otter_YOLO_Gimbal/<run>/weights/best.pt
```

Update `model_path` in `otter_usv_detector/config/detector.yaml` to point to the new weights.

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/collect_dataset.py` | Overhead pipeline — camera + USV teleportation, yaw+tilt model |
| `scripts/collect_dataset_gimbal.py` | Gimbal pipeline — FK-based camera pose, fixed USV, hard negatives |
| `scripts/annotator.py` | Pinhole projection, FK projection, YOLO label utilities |
| `scripts/verify_dataset.py` | Standalone visual QA tool |
| `launch/data_collection.launch` | Overhead collection simulation |
| `launch/data_collection_gimbal.launch` | Gimbal collection simulation |
| `config/collection_params.yaml` | Overhead sampling parameters |
| `config/collection_params_gimbal.yaml` | Gimbal sampling parameters |
| `config/lighting_params.yaml` | Gazebo sun/ambient lighting |
| `urdf/overhead_camera.urdf.xacro` | Virtual camera model (shared by both pipelines) |
| `worlds/open_water.world` | Open ocean Gazebo world (overhead pipeline) |
| `worlds/usv_uav_collect.world` | Test-environment world for gimbal pipeline (ocean + sky, no MRS plugins) |
