#!/usr/bin/env python3
"""
collect_dataset.py
──────────────────
ROS node that automates Otter USV dataset collection for YOLO OBB training.

Usage (via launch file):
    roslaunch dataset_acquisition data_collection.launch
    roslaunch dataset_acquisition data_collection.launch preview_only:=true

Usage (standalone after roscore + Gazebo are running):
    rosrun dataset_acquisition collect_dataset.py \
        _params_file:=/path/to/collection_params.yaml \
        _preview_only:=true

Sampling strategy
─────────────────
Camera XY and altitude are randomised independently. The Otter world position
is derived by back-projecting a uniformly sampled target pixel through the
tilted camera, so the AR marker centre lands at that pixel. This gives a
uniform spatial distribution of bounding boxes across the full image — including
corners and edges — which is critical for balanced YOLO OBB training.
"""

import os
import sys
import random
import shutil
import math
from typing import Optional

import numpy as np
import cv2
import rospy
import rospkg
import yaml
from scipy.spatial.transform import Rotation as ScipyRotation

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion

# local module (same scripts/ directory)
sys.path.insert(0, os.path.dirname(__file__))
from annotator import (compute_K, ar_marker_corners_local,
                        annotate, draw_obb, backproject_pixel_to_ground)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def yaw_tilt_to_quaternion(yaw_rad: float, tilt_rad: float = 0.0) -> Quaternion:
    """
    Build ROS Quaternion encoding Rz(yaw) · Ry(−tilt).
    tilt_rad=0 reproduces the original nadir-only behaviour.
    Why Ry(−tilt): the URDF cam_joint already applies Ry(+π/2) to achieve nadir;
    adding Ry(−tilt) to base_link rotates the nadir direction toward the yaw
    direction by tilt degrees — exactly the gimbal pitch action.
    """
    R = (ScipyRotation.from_euler('xyz', [0.0, 0.0, yaw_rad]) *
         ScipyRotation.from_euler('xyz', [0.0, -tilt_rad, 0.0]))
    q = R.as_quat()   # scipy: [x, y, z, w]
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def set_model_pose(set_state_srv, model_name: str,
                   x: float, y: float, z: float,
                   yaw_rad: float, tilt_rad: float = 0.0,
                   reference_frame: str = "world") -> bool:
    """Move a Gazebo model to (x, y, z) with orientation Rz(yaw)·Ry(−tilt)."""
    state = ModelState()
    state.model_name      = model_name
    state.reference_frame = reference_frame
    state.pose = Pose(
        position=Point(x=x, y=y, z=z),
        orientation=yaw_tilt_to_quaternion(yaw_rad, tilt_rad)
    )
    state.twist.linear.x  = 0.0; state.twist.linear.y  = 0.0; state.twist.linear.z  = 0.0
    state.twist.angular.x = 0.0; state.twist.angular.y = 0.0; state.twist.angular.z = 0.0
    resp = set_state_srv(state)
    return resp.success


def get_model_pose(get_state_srv, model_name: str):
    """Returns (pos_array[3], quat_array[4 xyzw]) or raises."""
    resp = get_state_srv(model_name, "world")
    if not resp.success:
        raise RuntimeError(f"get_model_state failed for '{model_name}'")
    p = resp.pose.position
    q = resp.pose.orientation
    return (np.array([p.x, p.y, p.z]),
            np.array([q.x, q.y, q.z, q.w]))


def split_and_move(src_images: list, src_labels: list,
                   dataset_dir: str,
                   train_r: float, val_r: float) -> None:
    """Shuffle and move collected files into train/val/test sub-directories."""
    indices = list(range(len(src_images)))
    random.shuffle(indices)

    n       = len(indices)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    splits = {
        "train": indices[:n_train],
        "val":   indices[n_train:n_train + n_val],
        "test":  indices[n_train + n_val:],
    }

    for split, idxs in splits.items():
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in idxs:
            shutil.move(src_images[i], img_dir)
            shutil.move(src_labels[i], lbl_dir)


def write_yolo_yaml(dataset_dir: str) -> None:
    cfg = {
        "path":  dataset_dir,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["otter_usv"],
    }
    out = os.path.join(dataset_dir, "otter_usv.yaml")
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    rospy.loginfo(f"[collect] YOLO dataset config → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main collector class
# ─────────────────────────────────────────────────────────────────────────────

class DatasetCollector:

    def __init__(self, params: dict, dataset_dir: str, preview_only: bool):
        self.p            = params
        self.dataset_dir  = dataset_dir
        self.preview_only = preview_only

        # Camera intrinsics
        cam_cfg        = params["camera"]
        self.img_w     = cam_cfg["width_px"]
        self.img_h     = cam_cfg["height_px"]
        self.K         = compute_K(self.img_w, self.img_h, cam_cfg["hfov_deg"])
        self.img_topic = cam_cfg["topic_image"]
        rospy.loginfo(f"[collect] Camera intrinsics K:\n{self.K}")

        # AR marker corners in Otter local frame
        ar = params["ar_marker"]
        self.corners_local = ar_marker_corners_local(
            ar["center_x_m"], ar["center_y_m"], ar["center_z_m"],
            ar["half_size_m"]
        )

        # Output directories
        self.tmp_dir     = os.path.join(dataset_dir, "_tmp")
        self.preview_dir = os.path.join(dataset_dir, "preview")
        os.makedirs(self.tmp_dir,     exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)

        # Persistent image subscriber
        self.bridge         = CvBridge()
        self.latest_img_msg = None
        rospy.Subscriber(self.img_topic, Image, self._img_cb, queue_size=1)

        # Gazebo services
        rospy.loginfo("[collect] Waiting for Gazebo services …")
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/get_model_state")
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        rospy.loginfo("[collect] Gazebo services ready.")

    def _img_cb(self, msg):
        self.latest_img_msg = msg

    # ── Image capture ──────────────────────────────────────────────────────

    def _capture_fresh_image(self, timeout_s: float = 8.0) -> Optional[np.ndarray]:
        """Flush buffer then wait for TWO consecutive new frames. Returns BGR or None.

        Gazebo's camera rendering pipeline can have a 1-frame lag after
        set_model_state: the first frame published after a teleport may have
        been rendered from the OLD camera position.  Waiting for the second
        frame guarantees the image reflects the new camera pose.
        """
        wait_start = rospy.Time.now()
        for _ in range(2):   # discard frame 1 (possibly stale), keep frame 2
            self.latest_img_msg = None
            while self.latest_img_msg is None and not rospy.is_shutdown():
                if (rospy.Time.now() - wait_start).to_sec() > timeout_s:
                    rospy.logwarn("[collect] Timeout waiting for fresh image.")
                    return None
                rospy.sleep(0.05)
        return self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding="bgr8")

    # ── Randomisation ──────────────────────────────────────────────────────

    def _random_sample_config(self) -> Optional[tuple]:
        """
        Randomise a complete camera + Otter configuration.

        Strategy:
          1. Camera XY, altitude, yaw, tilt — all independent.
          2. Sample a target pixel (u, v) uniformly within the image.
          3. Back-project that pixel through the tilted camera to find the
             world (x, y) where the AR marker centre should sit → Otter position.

        This guarantees the bounding box centre is uniformly distributed
        across the full image plane (corners, edges, and centre all equally
        likely), which is essential for balanced YOLO OBB training.

        Returns (cam_x, cam_y, cam_z, cam_yaw, cam_tilt,
                 otter_x, otter_y, otter_yaw)  or None on failure.
        """
        pose_cfg = self.p["pose"]
        ar       = self.p["ar_marker"]

        # 1. Camera pose — fully independent of Otter
        bound    = pose_cfg["camera_spawn_bounds_m"]
        cam_x    = random.uniform(-bound, bound)
        cam_y    = random.uniform(-bound, bound)
        cam_z    = random.uniform(pose_cfg["altitude_min_m"],
                                  pose_cfg["altitude_max_m"])
        cam_yaw  = random.uniform(math.radians(pose_cfg["cam_yaw_min_deg"]),
                                  math.radians(pose_cfg["cam_yaw_max_deg"]))
        cam_tilt = random.uniform(math.radians(pose_cfg["tilt_min_deg"]),
                                  math.radians(pose_cfg["tilt_max_deg"]))

        # 2. Sample target pixel — where the AR marker centre should appear
        m        = pose_cfg["target_margin_px"]
        u_target = random.uniform(m, self.img_w - m)
        v_target = random.uniform(m, self.img_h - m)

        # 3. Back-project → Otter world position
        cam_pos = np.array([cam_x, cam_y, cam_z])
        result  = backproject_pixel_to_ground(
            u_target, v_target, self.K,
            cam_pos, cam_yaw, cam_tilt,
            ground_z=ar["center_z_m"]   # z = 0.7 m (AR marker height)
        )
        if result is None:
            return None   # Ray pointed upward — safety guard

        otter_x, otter_y = result
        otter_yaw = random.uniform(0.0, 2.0 * math.pi)

        return cam_x, cam_y, cam_z, cam_yaw, cam_tilt, otter_x, otter_y, otter_yaw

    # ── Single random sample ───────────────────────────────────────────────

    def _collect_one(self) -> Optional[tuple]:
        """
        Randomise poses, capture image, annotate.
        Returns (bgr_image, label_str) or None if invisible / timeout.
        """
        otter_cfg = self.p["otter"]
        vis_cfg   = self.p["visibility"]

        config = self._random_sample_config()
        if config is None:
            return None
        cam_x, cam_y, cam_z, cam_yaw, cam_tilt, ot_x, ot_y, ot_yaw = config

        # 1. Move Otter
        set_model_pose(self._set_state, otter_cfg["model_name"],
                       ot_x, ot_y, 0.0, ot_yaw)

        # 2. Settle — allow water buoyancy to stabilise
        rospy.sleep(otter_cfg["settle_time_s"])

        # 3. Move camera with compound yaw + tilt quaternion
        set_model_pose(self._set_state, self.p["camera"]["model_name"],
                       cam_x, cam_y, cam_z, cam_yaw, cam_tilt)

        # 4. Let Gazebo renderer catch up
        rospy.sleep(0.5)

        # 5. Capture a fresh frame
        image = self._capture_fresh_image()
        if image is None:
            return None

        # 6. Query actual poses at capture time
        otter_pos, otter_quat = get_model_pose(self._get_state, otter_cfg["model_name"])
        cam_pos, _            = get_model_pose(self._get_state,
                                               self.p["camera"]["model_name"])

        # 7. Annotate
        label = annotate(
            self.corners_local,
            otter_pos, otter_quat,
            cam_pos,
            cam_yaw, cam_tilt,
            self.K,
            self.img_w, self.img_h,
            border_margin=vis_cfg["border_margin_px"],
            min_corners=vis_cfg["min_corners_in_frame"],
        )
        if label is None:
            return None
        return image, label

    # ── Structured sample (preview grid) ──────────────────────────────────

    def _collect_structured_one(self, spec: dict) -> Optional[tuple]:
        """
        Collect one sample from a deterministic spec dict (from preview_grid).
        Camera is fixed at world origin (x=0, y=0, z=alt) for reproducibility.
        Otter position is back-projected from the specified target pixel.

        Returns (bgr_image, label_str) or None on failure.
        """
        otter_cfg = self.p["otter"]
        vis_cfg   = self.p["visibility"]
        ar        = self.p["ar_marker"]

        cam_tilt  = math.radians(float(spec["tilt"]))
        cam_yaw   = math.radians(float(spec.get("cam_yaw", 0.0)))
        otter_yaw = math.radians(float(spec.get("otter_yaw", 0.0)))
        cam_z     = float(spec["alt"])
        cam_pos   = np.array([0.0, 0.0, cam_z])

        u_target  = float(spec["u_frac"]) * self.img_w
        v_target  = float(spec["v_frac"]) * self.img_h

        # Back-project target pixel → Otter world position
        result = backproject_pixel_to_ground(
            u_target, v_target, self.K,
            cam_pos, cam_yaw, cam_tilt,
            ground_z=ar["center_z_m"]
        )
        if result is None:
            rospy.logwarn(f"[collect] Back-projection failed for "
                          f"'{spec.get('label', '?')}'")
            return None

        otter_x, otter_y = result

        # Move Otter
        set_model_pose(self._set_state, otter_cfg["model_name"],
                       otter_x, otter_y, 0.0, otter_yaw)
        rospy.sleep(otter_cfg["settle_time_s"])

        # Move camera
        set_model_pose(self._set_state, self.p["camera"]["model_name"],
                       cam_pos[0], cam_pos[1], cam_pos[2], cam_yaw, cam_tilt)
        rospy.sleep(0.5)

        # Capture fresh frame
        image = self._capture_fresh_image()
        if image is None:
            return None

        # Query actual poses
        otter_pos, otter_quat = get_model_pose(self._get_state,
                                               otter_cfg["model_name"])
        cam_pos_actual, _     = get_model_pose(self._get_state,
                                               self.p["camera"]["model_name"])

        # Annotate
        label = annotate(
            self.corners_local,
            otter_pos, otter_quat,
            cam_pos_actual,
            cam_yaw, cam_tilt,
            self.K,
            self.img_w, self.img_h,
            border_margin=vis_cfg["border_margin_px"],
            min_corners=vis_cfg["min_corners_in_frame"],
        )
        if label is None:
            rospy.logwarn(f"[collect] USV not visible for "
                          f"'{spec.get('label', '?')}' — check tilt/altitude.")
            return None

        return image, label

    # ── Preview grid ───────────────────────────────────────────────────────

    def _collect_preview_grid(self):
        """
        Iterate through preview_grid from collection_params.yaml.
        Saves <label>_raw.jpg  and  <label>_obb.jpg  for each sample.

        Three groups are expected in the grid:
          Group 1 — Position sweep : USV at centre + 4 corners
          Group 2 — Tilt sweep     : nadir → 45°
          Group 3 — Altitude sweep : 5 m → 70 m
        """
        grid = self.p.get("preview_grid", [])
        if not grid:
            rospy.logwarn("[collect] No preview_grid defined — nothing to preview.")
            return

        passed = 0
        for i, spec in enumerate(grid):
            label_name = spec.get("label", f"sample_{i:02d}")
            rospy.loginfo(f"[collect] Preview {i+1}/{len(grid)}: {label_name}")

            result = self._collect_structured_one(spec)
            if result is None:
                rospy.logwarn(f"[collect] Skipping failed sample '{label_name}'.")
                continue

            image, yolo_label = result
            image_obb = draw_obb(image.copy(), yolo_label, self.img_w, self.img_h)

            # Overlay readable annotation text
            text = (f"alt={spec['alt']}m | tilt={spec['tilt']}deg"
                    f" | usv_yaw={spec.get('otter_yaw', 0)}deg")
            cv2.putText(image_obb, text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
                        cv2.LINE_AA)

            raw_path = os.path.join(self.preview_dir, f"{label_name}_raw.jpg")
            obb_path = os.path.join(self.preview_dir, f"{label_name}_obb.jpg")
            cv2.imwrite(raw_path, image)
            cv2.imwrite(obb_path, image_obb)
            passed += 1
            rospy.loginfo(f"[collect]   saved → {obb_path}")

        rospy.loginfo(
            f"[collect] Preview complete: {passed}/{len(grid)} samples"
            f" → {self.preview_dir}"
        )

    # ── Main run ───────────────────────────────────────────────────────────

    def run(self):
        # Wait for Otter and camera models to appear in Gazebo
        rospy.loginfo("[collect] Waiting for Gazebo models to fully spawn …")
        for model_name in [self.p["otter"]["model_name"],
                           self.p["camera"]["model_name"]]:
            while not rospy.is_shutdown():
                resp = self._get_state(model_name, "world")
                if resp.success:
                    break
                rospy.logwarn_throttle(
                    2.0, f"[collect] Still waiting for '{model_name}' …")
                rospy.sleep(1.0)
        rospy.loginfo("[collect] Models confirmed in simulation.")

        # Wait for camera plugin to come online — it can take 30-60 s after
        # Gazebo loads before the first image frame is published.
        rospy.loginfo("[collect] Waiting for camera to publish first frame …")
        try:
            rospy.wait_for_message(self.img_topic, Image, timeout=120.0)
            rospy.loginfo("[collect] Camera online.")
        except rospy.ROSException:
            rospy.logerr("[collect] Camera never published — check overhead_cam plugin.")
            return

        # ── Preview mode ──────────────────────────────────────────────────
        if self.preview_only:
            rospy.loginfo("[collect] === PREVIEW MODE (structured grid) ===")
            self._collect_preview_grid()
            return

        # ── Full collection ───────────────────────────────────────────────
        out_cfg = self.p["output"]
        n_total = out_cfg["n_samples"]
        fmt     = out_cfg["image_format"]
        jpeg_q  = out_cfg["jpeg_quality"]

        collected_imgs   = []
        collected_labels = []
        attempts = 0
        idx      = 0

        rospy.loginfo(f"[collect] Starting full collection — target {n_total} samples.")

        while idx < n_total and not rospy.is_shutdown():
            attempts += 1
            result = self._collect_one()
            if result is None:
                if attempts > n_total * 5:
                    rospy.logerr("[collect] Too many failed attempts — check setup.")
                    break
                continue

            image, label = result
            stem     = f"{idx:05d}"
            img_path = os.path.join(self.tmp_dir, f"{stem}.{fmt}")
            lbl_path = os.path.join(self.tmp_dir, f"{stem}.txt")

            if fmt == "jpg":
                cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            else:
                cv2.imwrite(img_path, image)
            with open(lbl_path, "w") as f:
                f.write(label + "\n")

            collected_imgs.append(img_path)
            collected_labels.append(lbl_path)
            idx += 1
            rospy.loginfo(
                f"[collect] Sample {idx}/{n_total}  (attempts so far: {attempts})")

        rospy.loginfo(f"[collect] Collected {idx} samples in {attempts} attempts.")

        if idx == 0:
            rospy.logerr("[collect] No samples collected — aborting.")
            return

        # Split, finalise, write YOLO config
        split_and_move(collected_imgs, collected_labels,
                       self.dataset_dir,
                       out_cfg["train_ratio"], out_cfg["val_ratio"])
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        write_yolo_yaml(self.dataset_dir)

        for split in ("train", "val", "test"):
            d = os.path.join(self.dataset_dir, "images", split)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            rospy.loginfo(f"[collect]   {split}: {n} images")

        rospy.loginfo("[collect] Dataset collection complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("collect_dataset", anonymous=False)

    params_file = rospy.get_param("~params_file", "")
    if not params_file:
        pkg = rospkg.RosPack().get_path("dataset_acquisition")
        params_file = os.path.join(pkg, "config", "collection_params.yaml")

    with open(params_file) as f:
        params = yaml.safe_load(f)

    dataset_dir = rospy.get_param("~dataset_dir", "")
    if not dataset_dir:
        pkg = rospkg.RosPack().get_path("dataset_acquisition")
        dataset_dir = os.path.join(pkg, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    preview_only = rospy.get_param("~preview_only", False)

    rospy.loginfo(f"[collect] Dataset directory : {dataset_dir}")
    rospy.loginfo(f"[collect] Params file       : {params_file}")
    rospy.loginfo(f"[collect] Preview only      : {preview_only}")

    rospy.loginfo("[collect] Waiting 5 s for Gazebo to stabilise …")
    rospy.sleep(5.0)

    collector = DatasetCollector(params, dataset_dir, preview_only)
    collector.run()


if __name__ == "__main__":
    main()
